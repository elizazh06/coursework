from pathlib import Path
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from datasets.base_dataset import BaseDataset

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _collect_files(root: Path, exts: set) -> Dict[str, Dict[str, Path]]:
    by_class = {}
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        items = {}
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                items[p.stem] = p
        if items:
            by_class[class_name] = items
    return by_class


def _collect_file_list(root: Path, exts: set) -> Dict[str, List[Path]]:
    by_class = {}
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
        )
        if files:
            by_class[class_dir.name] = files
    return by_class


def _collect_all_files(root: Path, exts: set) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _infer_class_name(path: Path, root: Path) -> str:
    rel_parts = path.relative_to(root).parts
    if len(rel_parts) >= 2:
        return rel_parts[-2]
    return path.parent.name if path.parent.name else "unknown"


def _pairs(vision_root: Path, audio_root: Path) -> List[Tuple[Path, Path, str]]:
    vis = _collect_files(vision_root, IMAGE_EXTS)
    aud = _collect_files(audio_root, AUDIO_EXTS)
    vis_lists = _collect_file_list(vision_root, IMAGE_EXTS)
    aud_lists = _collect_file_list(audio_root, AUDIO_EXTS)
    classes = sorted(set(vis.keys()) & set(aud.keys()))
    out = []
    for cls in classes:
        common = sorted(set(vis[cls].keys()) & set(aud[cls].keys()))
        if common:
            for stem in common:
                out.append((vis[cls][stem], aud[cls][stem], cls))
            continue

        # Fallback for datasets where paired media filenames differ by stem.
        # In that case we rely on deterministic sorted order within each class.
        vis_list = vis_lists.get(cls, [])
        aud_list = aud_lists.get(cls, [])
        n = min(len(vis_list), len(aud_list))
        for i in range(n):
            out.append((vis_list[i], aud_list[i], cls))

    if out:
        return out

    # Global fallback: recursively pair all files by deterministic order.
    # Useful when folder/class naming differs between audio and vision roots.
    vis_all = _collect_all_files(vision_root, IMAGE_EXTS)
    aud_all = _collect_all_files(audio_root, AUDIO_EXTS)
    n = min(len(vis_all), len(aud_all))
    for i in range(n):
        cls = _infer_class_name(vis_all[i], vision_root)
        out.append((vis_all[i], aud_all[i], cls))

    if not out:
        raise RuntimeError(
            "No paired samples found between vision and audio folders. "
            f"Found {len(_collect_all_files(vision_root, IMAGE_EXTS))} images and "
            f"{len(_collect_all_files(audio_root, AUDIO_EXTS))} audio files."
        )
    return out


def _split(
    items: List[Tuple[Path, Path, str]],
    split: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> List[Tuple[Path, Path, str]]:
    by_cls = {}
    for it in items:
        by_cls.setdefault(it[2], []).append(it)
    rng = random.Random(seed)
    selected = []
    for cls_items in by_cls.values():
        cls_items = list(cls_items)
        rng.shuffle(cls_items)
        n = len(cls_items)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        n_test = min(n_test, n)
        n_val = min(n_val, n - n_test)
        n_train = n - n_val - n_test
        if split == "train":
            selected.extend(cls_items[:n_train])
        elif split == "val":
            selected.extend(cls_items[n_train:n_train + n_val])
        else:
            selected.extend(cls_items[n_train + n_val:])
    return selected


class ADVANCEDataset(BaseDataset):
    def __init__(
        self,
        vision_root,
        audio_root,
        split="train",
        sample_rate=16000,
        n_mels=128,
        max_audio_seconds=5.0,
        image_size=224,
        val_ratio=0.1,
        test_ratio=0.1,
        split_seed=42,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
        **_,
    ):
        if torchaudio is None:
            raise ImportError("torchaudio is required for ADVANCEDataset.")

        self.split = split
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.max_audio_samples = int(float(max_audio_seconds) * self.sample_rate)
        self.image_tf = transforms.Compose(
            [
                transforms.Resize((int(image_size), int(image_size))),
                transforms.ToTensor(),
            ]
        )
        self.mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=self.n_mels,
        )

        all_pairs = _pairs(Path(vision_root).expanduser(), Path(audio_root).expanduser())
        classes = sorted({c for _, _, c in all_pairs})
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(self.class_to_idx)
        self.answer_to_idx = dict(self.class_to_idx)
        self.word_to_idx = {" ": 0}

        split_pairs = _split(
            all_pairs,
            split=split,
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            seed=int(split_seed),
        )
        index = [
            {
                "path": str(v),
                "audio_path": str(a),
                "label": self.class_to_idx[c],
                "class_name": c,
            }
            for v, a, c in split_pairs
        ]
        super().__init__(index, limit, shuffle_index, instance_transforms)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.image_tf(img)

    def _load_audio_feature(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.squeeze(0)
        if wav.numel() >= self.max_audio_samples:
            wav = wav[: self.max_audio_samples]
        else:
            wav = F.pad(wav, (0, self.max_audio_samples - wav.numel()))
        mel = self.mel_tf(wav.unsqueeze(0)).squeeze(0).transpose(0, 1)
        mel = torch.log(mel + 1e-6)
        return mel

    def __getitem__(self, ind):
        e = self._index[ind]
        video = self._load_image(e["path"]).unsqueeze(0)
        audio = self._load_audio_feature(e["audio_path"])
        label = torch.tensor(e["label"], dtype=torch.long)
        out = {
            "video": video,
            "audio": audio,
            "question": torch.zeros(1, dtype=torch.long),
            "label": label,
            "class_name": e["class_name"],
        }
        if self.instance_transforms:
            for k, fn in self.instance_transforms.items():
                if k in out and torch.is_tensor(out[k]):
                    out[k] = fn(out[k])
        return out

    @staticmethod
    def collate_batch(batch):
        return {
            "video": torch.stack([b["video"] for b in batch]),
            "audio": torch.stack([b["audio"] for b in batch]),
            "question": torch.stack([b["question"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]).long(),
            "class_name": [b["class_name"] for b in batch],
        }
