from pathlib import Path
import random
from typing import Dict, List, Tuple
import hashlib
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from datasets.base_dataset import BaseDataset
try:
    import torchaudio
except ImportError:
    torchaudio = None
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}

def _collect_files(root: Path, exts: set) -> Dict[str, Dict[str, Path]]:
    by_class = {}
    if not root.exists():
        raise FileNotFoundError(f'Path does not exist: {root}')
    for class_dir in sorted((p for p in root.iterdir() if p.is_dir())):
        class_name = class_dir.name
        items = {}
        for p in class_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                items[p.stem] = p
        if items:
            by_class[class_name] = items
    return by_class

def _collect_file_list(root: Path, exts: set) -> Dict[str, List[Path]]:
    by_class = {}
    if not root.exists():
        raise FileNotFoundError(f'Path does not exist: {root}')
    for class_dir in sorted((p for p in root.iterdir() if p.is_dir())):
        files = sorted((p for p in class_dir.rglob('*') if p.is_file() and p.suffix.lower() in exts))
        if files:
            by_class[class_dir.name] = files
    return by_class

def _collect_all_files(root: Path, exts: set) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f'Path does not exist: {root}')
    return sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts))

def _infer_class_name(path: Path, root: Path) -> str:
    rel_parts = path.relative_to(root).parts
    if len(rel_parts) >= 2:
        return rel_parts[-2]
    return path.parent.name if path.parent.name else 'unknown'

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
        vis_list = vis_lists.get(cls, [])
        aud_list = aud_lists.get(cls, [])
        n = min(len(vis_list), len(aud_list))
        for i in range(n):
            out.append((vis_list[i], aud_list[i], cls))
    if out:
        return out
    vis_all = _collect_all_files(vision_root, IMAGE_EXTS)
    aud_all = _collect_all_files(audio_root, AUDIO_EXTS)
    n = min(len(vis_all), len(aud_all))
    for i in range(n):
        cls = _infer_class_name(vis_all[i], vision_root)
        out.append((vis_all[i], aud_all[i], cls))
    if not out:
        raise RuntimeError(f'No paired samples found between vision and audio folders. Found {len(_collect_all_files(vision_root, IMAGE_EXTS))} images and {len(_collect_all_files(audio_root, AUDIO_EXTS))} audio files.')
    return out

def _split(items: List[Tuple[Path, Path, str]], split: str, val_ratio: float, test_ratio: float, seed: int) -> List[Tuple[Path, Path, str]]:
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
        if split == 'train':
            selected.extend(cls_items[:n_train])
        elif split == 'val':
            selected.extend(cls_items[n_train:n_train + n_val])
        else:
            selected.extend(cls_items[n_train + n_val:])
    return selected

class ADVANCEDataset(BaseDataset):

    def __init__(self, vision_root, audio_root, split='train', sample_rate=16000, n_mels=128, max_audio_seconds=5.0, image_size=224, val_ratio=0.1, test_ratio=0.1, split_seed=42, use_cache=True, prepare_cache_if_missing=True, rebuild_cache=False, cache_root=None, cache_prefix='advance_cache', limit=None, shuffle_index=False, instance_transforms=None, **_):
        if torchaudio is None:
            raise ImportError('torchaudio is required for ADVANCEDataset.')
        self.split = split
        self.vision_root = Path(vision_root).expanduser().resolve()
        self.audio_root = Path(audio_root).expanduser().resolve()
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.max_audio_samples = int(float(max_audio_seconds) * self.sample_rate)
        self.use_cache = bool(use_cache)
        self.prepare_cache_if_missing = bool(prepare_cache_if_missing)
        self.rebuild_cache = bool(rebuild_cache)
        self.image_tf = transforms.Compose([transforms.Resize((int(image_size), int(image_size))), transforms.ToTensor()])
        self.mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=1024, hop_length=256, n_mels=self.n_mels)
        self._cache_cfg = {'dataset': 'advance', 'sample_rate': self.sample_rate, 'n_mels': self.n_mels, 'max_audio_samples': self.max_audio_samples, 'image_size': int(image_size), 'val_ratio': float(val_ratio), 'test_ratio': float(test_ratio), 'split_seed': int(split_seed)}
        cfg_str = json.dumps(self._cache_cfg, sort_keys=True)
        cfg_hash = hashlib.md5(cfg_str.encode('utf-8')).hexdigest()[:12]
        base_cache_root = Path(cache_root).expanduser() if cache_root is not None else self.vision_root.parent / '.advance_cache'
        self.cache_dir = (base_cache_root / f'{cache_prefix}_{cfg_hash}' / split).resolve()
        all_pairs = _pairs(self.vision_root, self.audio_root)
        classes = sorted({c for (_, _, c) in all_pairs})
        self.class_to_idx = {c: i for (i, c) in enumerate(classes)}
        self.num_classes = len(self.class_to_idx)
        self.answer_to_idx = dict(self.class_to_idx)
        self.word_to_idx = {' ': 0}
        split_pairs = _split(all_pairs, split=split, val_ratio=float(val_ratio), test_ratio=float(test_ratio), seed=int(split_seed))
        if self.use_cache:
            index = self._build_or_load_cache_index(split_pairs)
        else:
            index = [{'path': str(v), 'audio_path': str(a), 'label': self.class_to_idx[c], 'class_name': c, 'cached': False} for (v, a, c) in split_pairs]
        super().__init__(index, limit, shuffle_index, instance_transforms)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        return self.image_tf(img)

    def _load_audio_feature(self, path: str) -> torch.Tensor:
        (wav, sr) = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.squeeze(0)
        if wav.numel() >= self.max_audio_samples:
            wav = wav[:self.max_audio_samples]
        else:
            wav = F.pad(wav, (0, self.max_audio_samples - wav.numel()))
        mel = self.mel_tf(wav.unsqueeze(0)).squeeze(0).transpose(0, 1)
        mel = torch.log(mel + 1e-06)
        return mel

    def _build_or_load_cache_index(self, split_pairs):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.cache_dir / 'index.json'
        meta_path = self.cache_dir / 'meta.json'
        should_rebuild = self.rebuild_cache
        if not should_rebuild and index_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                if meta.get('cache_cfg') == self._cache_cfg:
                    return json.loads(index_path.read_text(encoding='utf-8'))
                should_rebuild = True
            except (json.JSONDecodeError, OSError):
                should_rebuild = True
        elif not index_path.exists() or not meta_path.exists():
            should_rebuild = True
        if should_rebuild and (not self.prepare_cache_if_missing):
            raise RuntimeError(f'ADVANCE cache is missing/invalid at {self.cache_dir}. Enable `prepare_cache_if_missing` or run one training pass to build it.')
        index = []
        for (i, (v, a, c)) in enumerate(tqdm(split_pairs, desc=f'Prepare ADVANCE cache [{self.split}]')):
            video = self._load_image(str(v)).unsqueeze(0)
            audio = self._load_audio_feature(str(a))
            payload = {'video': video, 'audio': audio, 'label': torch.tensor(self.class_to_idx[c], dtype=torch.long), 'class_name': c}
            cache_path = self.cache_dir / f'{i:08d}.pt'
            torch.save(payload, cache_path)
            index.append({'path': str(cache_path), 'label': self.class_to_idx[c], 'class_name': c, 'cached': True})
        index_path.write_text(json.dumps(index, ensure_ascii=True), encoding='utf-8')
        meta = {'cache_cfg': self._cache_cfg, 'cache_dir': str(self.cache_dir), 'split': self.split}
        meta_path.write_text(json.dumps(meta, ensure_ascii=True), encoding='utf-8')
        return index

    def __getitem__(self, ind):
        e = self._index[ind]
        if e.get('cached', False):
            item = torch.load(e['path'], map_location='cpu', weights_only=False)
            video = item['video']
            audio = item['audio']
            label = item['label'].long()
            class_name = item['class_name']
        else:
            video = self._load_image(e['path']).unsqueeze(0)
            audio = self._load_audio_feature(e['audio_path'])
            label = torch.tensor(e['label'], dtype=torch.long)
            class_name = e['class_name']
        out = {'video': video, 'audio': audio, 'question': torch.zeros(1, dtype=torch.long), 'label': label, 'class_name': class_name}
        if self.instance_transforms:
            for (k, fn) in self.instance_transforms.items():
                if k in out and torch.is_tensor(out[k]):
                    out[k] = fn(out[k])
        return out

    @staticmethod
    def collate_batch(batch):
        return {'video': torch.stack([b['video'] for b in batch]), 'audio': torch.stack([b['audio'] for b in batch]), 'question': torch.stack([b['question'] for b in batch]), 'label': torch.stack([b['label'] for b in batch]).long(), 'class_name': [b['class_name'] for b in batch]}
