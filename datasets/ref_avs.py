"""RefAVSDataset — Ref-AVS-Bench dataset loaded from Kaggle.

Expected directory layout (all paths are configurable via YAML params):

  <data_root>/
    ├── v2_data/                       # video frames + GT masks per UID
    │   └── <uid>/
    │       ├── frames/
    │       │   ├── 00001.jpg
    │       │   └── 00002.jpg  (num_frames total)
    │       └── gt_masks/
    │           ├── 00001.png
    │           └── 00002.png
    ├── audio_features/                # pre-extracted audio features
    │   └── <uid>.npy                  # float32, shape [T_frames, 128]
    ├── text_features/                 # pre-extracted BERT text features
    │   └── <uid>.npy                  # float32, shape [T_tokens, 768]
    └── metadata/
        ├── train.csv                  # columns: uid, text_query  (or .txt with uid)
        ├── val.csv
        └── test.csv

Alternative split format: plain .txt with one UID per line (text_query left empty).
Alternative audio/text source: raw .wav + text_query string with on-the-fly encoding
(set use_preextracted=False, requires torchaudio + transformers[bert]).
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from datasets.base_dataset import BaseDataset

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_NPY_EXTS = {".npy", ".npz"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sorted_frames(uid_dir: Path, frames_subdir: str) -> List[Path]:
    frames_root = uid_dir / frames_subdir
    if not frames_root.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_root}")
    return sorted(p for p in frames_root.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def _sorted_masks(uid_dir: Path, masks_subdir: str) -> List[Path]:
    masks_root = uid_dir / masks_subdir
    if not masks_root.exists():
        return []
    return sorted(p for p in masks_root.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def _load_npy_feat(path: Path) -> torch.Tensor:
    arr = np.load(str(path))
    return torch.from_numpy(arr).float()


def _read_split_file(path: Path) -> List[Dict]:
    """Parse split file (CSV with uid[,text_query] or plain TXT with uid)."""
    records: List[Dict] = []
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            records.append(
                {"uid": str(item["uid"]), "text_query": item.get("text_query", "")}
            )
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    {"uid": row["uid"].strip(), "text_query": row.get("text_query", "").strip()}
                )
    else:
        # Plain .txt: one UID per line, optional <TAB>text_query
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            records.append(
                {"uid": parts[0].strip(), "text_query": parts[1].strip() if len(parts) > 1 else ""}
            )
    return records


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class RefAVSDataset(BaseDataset):
    """PyTorch dataset for Ref-AVS-Bench.

    Each item returned:
        uid          (str)         – sample identifier
        frames       (T, C, H, W)  – normalised RGB frames
        audio_feat   (T_a, 128)    – pre-extracted audio features
        text_feat    (T_t, 768)    – pre-extracted BERT text features
        masks        (T, H_m, W_m) – binary ground-truth masks (float 0/1)
        text_query   (str)         – referring expression
        label        (scalar long) – dummy 0 (pipeline compatibility)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        # sub-directory names inside data_root
        video_dir: str = "v2_data",
        frames_subdir: str = "frames",
        masks_subdir: str = "gt_masks",
        audio_feat_dir: str = "audio_features",
        text_feat_dir: str = "text_features",
        metadata_dir: str = "metadata",
        # split file name pattern: {split}.csv / {split}.txt / {split}.json
        split_filename: Optional[str] = None,
        # pre-processing
        num_frames: int = 5,
        mask_size: int = 256,
        image_size: int = 256,
        # text / audio fallback
        use_preextracted_audio: bool = True,
        use_preextracted_text: bool = True,
        audio_sample_rate: int = 16000,
        audio_n_mels: int = 128,
        bert_model: str = "bert-base-uncased",
        # base-dataset params
        limit: Optional[int] = None,
        shuffle_index: bool = False,
        instance_transforms=None,
        **_,
    ):
        self.split = split
        self.data_root = Path(data_root).expanduser().resolve()
        self.video_root = self.data_root / video_dir
        self.frames_subdir = frames_subdir
        self.masks_subdir = masks_subdir
        self.audio_feat_root = self.data_root / audio_feat_dir
        self.text_feat_root = self.data_root / text_feat_dir

        self.num_frames = int(num_frames)
        self.mask_size = int(mask_size)

        self.use_preextracted_audio = bool(use_preextracted_audio)
        self.use_preextracted_text = bool(use_preextracted_text)

        self.img_tf = transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.mask_resize = transforms.Resize(
            (int(mask_size), int(mask_size)),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        # ---- lazy imports for fallback encoders ----
        self._mel_tf = None
        self._bert_tokenizer = None
        self._bert_model = None
        if not use_preextracted_audio:
            self._init_audio_encoder(audio_sample_rate, audio_n_mels)
        if not use_preextracted_text:
            self._init_text_encoder(bert_model)

        # ---- build index ----
        records = self._load_records(metadata_dir, split_filename)
        index = self._build_index(records)
        super().__init__(index, limit, shuffle_index, instance_transforms)

    # ------------------------------------------------------------------
    # initialisation helpers
    # ------------------------------------------------------------------

    def _init_audio_encoder(self, sample_rate: int, n_mels: int):
        try:
            import torchaudio
            self._torchaudio = torchaudio
            self._mel_tf = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=n_mels
            )
            self._audio_sr = sample_rate
        except ImportError:
            raise ImportError(
                "torchaudio is required when use_preextracted_audio=False. "
                "Install via: pip install torchaudio"
            )

    def _init_text_encoder(self, bert_model: str):
        try:
            from transformers import BertTokenizer, BertModel
            self._bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
            self._bert_model = BertModel.from_pretrained(bert_model).eval()
        except ImportError:
            raise ImportError(
                "transformers is required when use_preextracted_text=False. "
                "Install via: pip install transformers"
            )

    # ------------------------------------------------------------------
    # split / metadata loading
    # ------------------------------------------------------------------

    def _load_records(self, metadata_dir: str, split_filename: Optional[str]) -> List[Dict]:
        if split_filename is not None:
            path = self.data_root / split_filename
            if not path.exists():
                path = self.data_root / metadata_dir / split_filename
        else:
            meta_dir = self.data_root / metadata_dir
            candidates = [
                meta_dir / f"{self.split}.csv",
                meta_dir / f"{self.split}.json",
                meta_dir / f"{self.split}.txt",
                self.data_root / f"{self.split}.csv",
                self.data_root / f"{self.split}.json",
                self.data_root / f"{self.split}.txt",
            ]
            path = next((p for p in candidates if p.exists()), None)

        if path is None or not path.exists():
            warnings.warn(
                f"No split file found for '{self.split}' in {self.data_root}. "
                "Scanning video_root directory for UID folders.",
                UserWarning,
                stacklevel=3,
            )
            return self._scan_uid_dirs()

        return _read_split_file(path)

    def _scan_uid_dirs(self) -> List[Dict]:
        """Fallback: enumerate all UID sub-dirs in video_root."""
        if not self.video_root.exists():
            raise FileNotFoundError(f"video_root does not exist: {self.video_root}")
        uids = sorted(
            p.name for p in self.video_root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        return [{"uid": u, "text_query": ""} for u in uids]

    # ------------------------------------------------------------------
    # index construction
    # ------------------------------------------------------------------

    def _build_index(self, records: List[Dict]) -> List[Dict]:
        index = []
        for rec in records:
            uid = rec["uid"]
            uid_dir = self.video_root / uid

            frame_paths = _sorted_frames(uid_dir, self.frames_subdir)
            if not frame_paths:
                warnings.warn(f"No frames found for UID {uid}, skipping.", stacklevel=3)
                continue

            mask_paths = _sorted_masks(uid_dir, self.masks_subdir)

            # audio feature path
            audio_feat_path = self._find_feat_file(self.audio_feat_root, uid)
            # text feature path
            text_feat_path = self._find_feat_file(self.text_feat_root, uid)

            index.append({
                "path": str(uid_dir),          # required by BaseDataset
                "label": 0,                    # dummy
                "uid": uid,
                "text_query": rec.get("text_query", ""),
                "frame_paths": [str(p) for p in frame_paths],
                "mask_paths": [str(p) for p in mask_paths],
                "audio_feat_path": str(audio_feat_path) if audio_feat_path else None,
                "text_feat_path": str(text_feat_path) if text_feat_path else None,
            })
        return index

    @staticmethod
    def _find_feat_file(feat_root: Path, uid: str) -> Optional[Path]:
        for ext in (".npy", ".npz", ".pt"):
            p = feat_root / f"{uid}{ext}"
            if p.exists():
                return p
        return None

    # ------------------------------------------------------------------
    # loading helpers
    # ------------------------------------------------------------------

    def _load_frames(self, paths: List[str]) -> torch.Tensor:
        """Load and preprocess up to num_frames images → [T, C, H, W]."""
        selected = self._select_frames(paths)
        tensors = [self.img_tf(Image.open(p).convert("RGB")) for p in selected]
        return torch.stack(tensors)   # [T, C, H, W]

    def _load_masks(self, paths: List[str]) -> torch.Tensor:
        """Load GT masks → [T, H_m, W_m] float32 binary."""
        if not paths:
            return torch.zeros(self.num_frames, self.mask_size, self.mask_size)
        selected = self._select_frames(paths)
        masks = []
        for p in selected:
            img = Image.open(p).convert("L")
            img = self.mask_resize(img)
            m = torch.from_numpy(np.array(img, dtype=np.float32))
            m = (m > 127).float()
            masks.append(m)
        return torch.stack(masks)   # [T, H_m, W_m]

    def _select_frames(self, paths: List[str]) -> List[str]:
        """Sample exactly num_frames from the path list."""
        n = len(paths)
        if n == self.num_frames:
            return paths
        if n > self.num_frames:
            idxs = [int(round(i * (n - 1) / (self.num_frames - 1))) for i in range(self.num_frames)]
            return [paths[i] for i in idxs]
        # pad with last frame
        return paths + [paths[-1]] * (self.num_frames - n)

    def _load_audio_feat(self, entry: Dict) -> torch.Tensor:
        if self.use_preextracted_audio and entry["audio_feat_path"]:
            feat = _load_npy_feat(Path(entry["audio_feat_path"]))
            # ensure shape [T_a, 128]
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            return feat
        # fallback: load raw wav from <uid_dir>/audio.wav
        uid_dir = Path(entry["path"])
        wav_candidates = list(uid_dir.glob("audio.*"))
        if not wav_candidates:
            return torch.zeros(self.num_frames, 128)
        wav_path = wav_candidates[0]
        wav, sr = self._torchaudio.load(str(wav_path))
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self._audio_sr:
            wav = self._torchaudio.functional.resample(wav, sr, self._audio_sr)
        mel = self._mel_tf(wav).squeeze(0).transpose(0, 1)   # [T, n_mels]
        mel = torch.log(mel + 1e-6)
        return mel

    def _load_text_feat(self, entry: Dict) -> torch.Tensor:
        if self.use_preextracted_text and entry["text_feat_path"]:
            feat = _load_npy_feat(Path(entry["text_feat_path"]))
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            return feat
        # fallback: BERT encoding of text_query
        query = entry.get("text_query", "") or "audio visual segmentation"
        with torch.no_grad():
            enc = self._bert_tokenizer(
                query, return_tensors="pt", truncation=True, max_length=77
            )
            out = self._bert_model(**enc)
            feat = out.last_hidden_state.squeeze(0)   # [T_t, 768]
        return feat.float()

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, ind: int) -> Dict:
        entry = self._index[ind]

        frames = self._load_frames(entry["frame_paths"])       # [T, C, H, W]
        masks = self._load_masks(entry["mask_paths"])          # [T, H_m, W_m]
        audio_feat = self._load_audio_feat(entry)              # [T_a, 128]
        text_feat = self._load_text_feat(entry)                # [T_t, 768]

        out = {
            "uid": entry["uid"],
            "frames": frames,
            "audio_feat": audio_feat,
            "text_feat": text_feat,
            "masks": masks,
            "text_query": entry.get("text_query", ""),
            "label": torch.tensor(0, dtype=torch.long),
        }

        if self.instance_transforms:
            for k, fn in self.instance_transforms.items():
                if k in out and torch.is_tensor(out[k]):
                    out[k] = fn(out[k])
        return out

    # ------------------------------------------------------------------
    # collate
    # ------------------------------------------------------------------

    @staticmethod
    def collate_batch(batch: List[Dict]) -> Dict:
        """Collate with variable-length audio/text feature padding."""
        # fixed-shape tensors
        frames = torch.stack([b["frames"] for b in batch])     # [B, T, C, H, W]
        masks = torch.stack([b["masks"] for b in batch])       # [B, T, H_m, W_m]
        labels = torch.stack([b["label"] for b in batch])

        # variable-length: pad to max length in batch
        def pad_feat(key: str, last_dim: int) -> torch.Tensor:
            feats = [b[key] for b in batch]
            max_len = max(f.size(0) for f in feats)
            padded = torch.zeros(len(feats), max_len, last_dim)
            for i, f in enumerate(feats):
                padded[i, :f.size(0)] = f
            return padded

        audio_feat = pad_feat("audio_feat", 128)    # [B, T_a, 128]
        text_feat = pad_feat("text_feat", 768)       # [B, T_t, 768]

        return {
            "uid": [b["uid"] for b in batch],
            "frames": frames,
            "audio_feat": audio_feat,
            "text_feat": text_feat,
            "masks": masks,
            "text_query": [b["text_query"] for b in batch],
            "label": labels,
        }
