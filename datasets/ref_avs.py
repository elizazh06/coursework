"""RefAVSDataset — Ref-AVS-Bench dataset.

Actual directory layout (from dataset_refavs.py in GeWu-Lab/Ref-AVS):

  <data_root>/
    ├── metadata.csv              # vid, uid, fid, exp, split
    ├── media/
    │   └── <vid>/
    │       ├── audio.wav
    │       └── frames/
    │           ├── 0.jpg
    │           ├── 1.jpg
    │           └── ...           (0-indexed, frame_n total)
    └── gt_mask/
        └── <vid>/
            └── fid_<fid>/
                ├── 00000.png
                ├── 00001.png
                └── ...           (00000..0000{frame_n-1}.png)

metadata.csv columns:
  vid   — video identifier  (e.g. "abc123")
  uid   — annotation id, encodes vid + fid  (e.g. "abc123_0_1")
  fid   — frame / clip annotation id (int)
  exp   — referring text expression
  split — train | val | test_s | test_u | test_n

Splits exposed by this class:
  "train"  → rows with split == "train"
  "val"    → rows with split == "val"
  "test"   → rows with split == "test_s"  (seen; primary benchmark split)
  "test_s" → same as "test"
  "test_u" → unseen split
  "test_n" → null (no visible sound source) split

Feature caching:
  On first run with use_cache=True (default) all text and audio features are
  pre-extracted and saved to <cache_root>/<uid>.pt. Subsequent runs load them
  from disk — this reduces per-sample latency from ~600 ms to ~2 ms and
  speeds up a Kaggle epoch from ~4 h to ~20-30 min.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from datasets.base_dataset import BaseDataset

# Lazy imports — only required when encoders are initialised
try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

# map user-facing split names → metadata.csv split column values
_SPLIT_MAP = {
    "train": "train",
    "val": "val",
    "test": "test_s",
    "test_s": "test_s",
    "test_u": "test_u",
    "test_n": "test_n",
}


class RefAVSDataset(BaseDataset):
    """PyTorch dataset for Ref-AVS-Bench.

    Each item:
        uid          (str)          – annotation uid from metadata
        frames       (T, C, H, W)   – normalised RGB frames (float32)
        audio_feat   (T, 128)       – mel-based audio features (float32)
        text_feat    (T_t, 768)     – DistilRoBERTa text embedding (float32)
        masks        (T, H_m, W_m)  – binary GT masks (float32, 0/1)
        text_query   (str)          – raw referring expression
        label        (scalar long)  – dummy 0 (pipeline compatibility)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        # processing params
        frame_n: int = 10,
        mask_size: int = 256,
        image_size: int = 256,
        # audio params (MelSpectrogram for 128-dim features)
        sample_rate: int = 16000,
        # text model (DistilRoBERTa, same as official code)
        text_model: str = "distilbert/distilroberta-base",
        text_max_len: int = 25,
        # feature caching — pre-extract text+audio once, load from disk afterward
        use_cache: bool = True,
        cache_root: Optional[str] = None,
        rebuild_cache: bool = False,
        # base-dataset
        limit: Optional[int] = None,
        shuffle_index: bool = False,
        instance_transforms=None,
        **_,
    ):
        self.split = split
        self.data_root = Path(data_root).expanduser().resolve()
        self.media_dir = self.data_root / "media"
        self.mask_dir = self.data_root / "gt_mask"

        self.frame_n = int(frame_n)
        self.mask_size = int(mask_size)
        self.image_size = int(image_size)
        self.sample_rate = int(sample_rate)
        self.text_max_len = int(text_max_len)
        self.text_model_name = text_model

        self.use_cache = bool(use_cache)
        self.rebuild_cache = bool(rebuild_cache)

        # cache dir: unique per (text_model, text_max_len, sample_rate, frame_n)
        cfg_hash = hashlib.md5(
            json.dumps({
                "text_model": text_model,
                "text_max_len": text_max_len,
                "sample_rate": sample_rate,
                "frame_n": frame_n,
            }, sort_keys=True).encode()
        ).hexdigest()[:10]
        base_cache = (
            Path(cache_root).expanduser()
            if cache_root else self.data_root / ".ref_avs_cache"
        )
        self.cache_dir = base_cache / f"{cfg_hash}_{split}"

        self.img_tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # ── audio encoder (MelSpectrogram, 128 mel bins) ──────────────────
        if not _TORCHAUDIO_AVAILABLE:
            raise ImportError("torchaudio is required. pip install torchaudio")
        self._mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
        )

        # ── text encoder (DistilRoBERTa) ───────────────────────────────────
        try:
            from transformers import AutoTokenizer, AutoModel, logging as hf_logging
            hf_logging.set_verbosity_error()
            self._tokenizer = AutoTokenizer.from_pretrained(text_model)
            self._text_encoder = AutoModel.from_pretrained(text_model).eval()
        except ImportError:
            raise ImportError("transformers is required. pip install transformers")

        # ── build index + optionally pre-build cache ───────────────────────
        index = self._build_index()
        if self.use_cache:
            index = self._ensure_cache(index)
        super().__init__(index, limit, shuffle_index, instance_transforms)

    # ------------------------------------------------------------------
    # index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> List[Dict]:
        meta_path = self.data_root / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.csv not found at {meta_path}.\n"
                f"Expected layout: {self.data_root}/metadata.csv"
            )

        df = pd.read_csv(meta_path, header=0)
        csv_split = _SPLIT_MAP.get(self.split)
        if csv_split is None:
            raise ValueError(
                f"Unknown split '{self.split}'. Valid: {list(_SPLIT_MAP.keys())}"
            )

        df = df[df["split"] == csv_split].reset_index(drop=True)
        if len(df) == 0:
            warnings.warn(
                f"No rows found for split '{self.split}' (csv='{csv_split}') "
                f"in {meta_path}.",
                UserWarning, stacklevel=3,
            )

        index = []
        for _, row in df.iterrows():
            vid = str(row["vid"])
            uid = str(row["uid"])
            fid = str(row["fid"])
            exp = str(row["exp"])

            frames_dir = self.media_dir / vid / "frames"
            audio_path = self.media_dir / vid / "audio.wav"
            mask_dir = self.mask_dir / vid / f"fid_{fid}"

            if not frames_dir.exists():
                warnings.warn(
                    f"Frames dir missing: {frames_dir}. Skipping uid={uid}.",
                    stacklevel=3,
                )
                continue

            index.append({
                "path": str(frames_dir),   # required by BaseDataset
                "label": 0,
                "uid": uid,
                "vid": vid,
                "fid": fid,
                "text_query": exp,
                "audio_path": str(audio_path),
                "mask_dir": str(mask_dir),
                "cache_path": None,   # filled in _ensure_cache
            })
        return index

    # ------------------------------------------------------------------
    # feature cache
    # ------------------------------------------------------------------

    def _ensure_cache(self, index: List[Dict]) -> List[Dict]:
        """Pre-extract text + audio features and save to disk.

        Only runs on first call (or when rebuild_cache=True). All subsequent
        epochs load features in ~2 ms per sample instead of ~600 ms.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.cache_dir / "meta.json"

        expected_meta = {
            "text_model": self.text_model_name,
            "text_max_len": self.text_max_len,
            "sample_rate": self.sample_rate,
            "frame_n": self.frame_n,
            "split": self.split,
        }

        already_built = False
        if not self.rebuild_cache and meta_path.exists():
            try:
                saved_meta = json.loads(meta_path.read_text())
                already_built = saved_meta == expected_meta
            except Exception:
                pass

        # assign cache paths regardless (used in __getitem__)
        for entry in index:
            entry["cache_path"] = str(self.cache_dir / f"{entry['uid']}.pt")

        if already_built:
            return index

        print(f"[RefAVSDataset] Building feature cache for split='{self.split}' "
              f"→ {self.cache_dir}")

        # Move text encoder to GPU if available for faster extraction
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._text_encoder = self._text_encoder.to(device)

        for entry in tqdm(index, desc=f"Caching {self.split} features"):
            cache_pt = Path(entry["cache_path"])
            if cache_pt.exists() and not self.rebuild_cache:
                continue
            audio_feat = self._compute_audio_feat(entry["audio_path"])
            text_feat = self._compute_text_feat(entry["text_query"], device)
            torch.save(
                {"audio_feat": audio_feat.cpu(), "text_feat": text_feat.cpu()},
                cache_pt,
            )

        meta_path.write_text(json.dumps(expected_meta))
        # Move text encoder back to CPU to free GPU memory before training
        self._text_encoder = self._text_encoder.cpu()
        print(f"[RefAVSDataset] Cache built: {len(index)} entries.")
        return index

    # ------------------------------------------------------------------
    # raw feature computation (used during cache build or when cache=False)
    # ------------------------------------------------------------------

    def _compute_audio_feat(self, audio_path: str) -> torch.Tensor:
        """Load .wav → MelSpectrogram pooled to [frame_n, 128]."""
        wav_path = Path(audio_path)
        if not wav_path.exists():
            return torch.zeros(self.frame_n, 128)

        wav, sr = torchaudio.load(str(wav_path))
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        mel = self._mel_tf(wav).squeeze(0)          # [128, T_mel]
        mel = torch.log(mel + 1e-6)

        t_mel = mel.size(1)
        if t_mel >= self.frame_n:
            segments = torch.chunk(mel, self.frame_n, dim=1)
            feat = torch.stack([s.mean(dim=1) for s in segments])  # [T, 128]
        else:
            feat = F.adaptive_avg_pool1d(
                mel.unsqueeze(0), self.frame_n
            ).squeeze(0).T
        return feat   # [frame_n, 128]

    def _compute_text_feat(self, exp: str, device: str = "cpu") -> torch.Tensor:
        """Encode text with DistilRoBERTa → [T_t, 768]."""
        inputs = self._tokenizer(
            exp,
            max_length=self.text_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feat = self._text_encoder(**inputs).last_hidden_state  # [1, T_t, 768]
        return feat.squeeze(0).float()   # [T_t, 768]

    # ------------------------------------------------------------------
    # frame / mask loading
    # ------------------------------------------------------------------

    def _load_frames(self, frames_dir: str) -> torch.Tensor:
        tensors = []
        for i in range(self.frame_n):
            path = Path(frames_dir) / f"{i}.jpg"
            if not path.exists():
                path = Path(frames_dir) / f"{i:05d}.jpg"
            if path.exists():
                img = Image.open(path).convert("RGB")
            else:
                img = Image.new("RGB", (self.image_size, self.image_size))
                warnings.warn(f"Frame not found: {Path(frames_dir) / f'{i}.jpg'}")
            tensors.append(self.img_tf(img))
        return torch.stack(tensors)   # [T, C, H, W]

    def _load_masks(self, mask_dir: str) -> torch.Tensor:
        masks = []
        for i in range(self.frame_n):
            path = Path(mask_dir) / f"{i:05d}.png"
            if path.exists():
                mask_cv2 = cv2.imread(str(path))
                mask_cv2 = cv2.resize(mask_cv2, (self.mask_size, self.mask_size))
                mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)
                m = torch.as_tensor(mask_cv2 > 0, dtype=torch.float32)
            else:
                m = torch.zeros(self.mask_size, self.mask_size)
            masks.append(m)
        return torch.stack(masks)   # [T, H_m, W_m]

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, ind: int) -> Dict:
        entry = self._index[ind]

        frames = self._load_frames(entry["path"])    # [T, C, H, W]
        masks = self._load_masks(entry["mask_dir"])  # [T, H_m, W_m]

        # Load features from cache or compute on the fly
        if self.use_cache and entry.get("cache_path"):
            cached = torch.load(entry["cache_path"], map_location="cpu",
                                weights_only=True)
            audio_feat = cached["audio_feat"]   # [T, 128]
            text_feat = cached["text_feat"]     # [T_t, 768]
        else:
            audio_feat = self._compute_audio_feat(entry["audio_path"])
            text_feat = self._compute_text_feat(entry["text_query"])

        out = {
            "uid": entry["uid"],
            "frames": frames,
            "audio_feat": audio_feat,
            "text_feat": text_feat,
            "masks": masks,
            "text_query": entry["text_query"],
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
        frames = torch.stack([b["frames"] for b in batch])       # [B, T, C, H, W]
        masks = torch.stack([b["masks"] for b in batch])         # [B, T, H_m, W_m]
        audio_feat = torch.stack([b["audio_feat"] for b in batch])  # [B, T, 128]
        labels = torch.stack([b["label"] for b in batch])

        # text features may differ in T_t — pad to max
        text_feats = [b["text_feat"] for b in batch]
        max_tt = max(f.size(0) for f in text_feats)
        text_feat_padded = torch.zeros(len(batch), max_tt, 768)
        for i, f in enumerate(text_feats):
            text_feat_padded[i, :f.size(0)] = f

        return {
            "uid": [b["uid"] for b in batch],
            "frames": frames,
            "audio_feat": audio_feat,
            "text_feat": text_feat_padded,
            "masks": masks,
            "text_query": [b["text_query"] for b in batch],
            "label": labels,
        }
