import ast
import json
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from tqdm.auto import tqdm

from datasets.base_dataset import BaseDataset
from utils.io import download_file, read_json, write_json
from utils.media import extract_wav_from_video, mel_from_wav_path, trim_pad_time


def _tokenize(item):
    q = item["question_content"].rstrip().split(" ")
    if q and q[-1].endswith("?"):
        q[-1] = q[-1][:-1]
    tv = item.get("templ_values", "[]")
    try:
        tv = ast.literal_eval(tv) if isinstance(tv, str) else tv
    except (ValueError, SyntaxError):
        tv = []
    j = 0
    for i, t in enumerate(q):
        if "<" in t and j < len(tv):
            q[i] = str(tv[j])
            j += 1
    return [w.lower() for w in q if w]


def _vocabs(rows):
    w2i, a2i = {" ": 0}, {}
    for it in rows:
        for t in _tokenize(it):
            w2i.setdefault(t, len(w2i))
        a = it.get("answer", it.get("anser", "unknown"))
        a2i.setdefault(a, len(a2i))
    return w2i, a2i


def _frames(path, L):
    p = Path(path)
    if not p.exists():
        return torch.zeros(L, 3, 224, 224)
    vr = VideoReader(str(p), ctx=cpu(0))
    if len(vr) == 0:
        return torch.zeros(L, 3, 224, 224)
    ix = torch.linspace(0, len(vr) - 1, L).long().tolist()
    fr = vr.get_batch(ix)
    return torch.tensor(fr.asnumpy()).permute(0, 3, 1, 2).float() / 255.0


def _pack(item, idx, root, mx, st, strict, uf, dcfg, fad, fvd, w2i, a2i):
    vid = item["video_id"]
    vp, ap = root / "video" / f"{vid}.mp4", root / "audio" / f"{vid}.wav"
    base = str((dcfg.get("media") or {}).get("video_base_url", "")).rstrip("/")
    au = vi = None
    if uf:
        f1, f2 = fad / f"{vid}.npy", fvd / f"{vid}.npy"
        if f1.exists():
            au = trim_pad_time(torch.from_numpy(np.load(f1)).float(), mx, st)
        if f2.exists():
            vi = trim_pad_time(torch.from_numpy(np.load(f2)).float(), mx, st)
    ra = rv = False
    if au is None or vi is None:
        if not vp.exists():
            if not base:
                if strict:
                    raise FileNotFoundError(vp)
            else:
                download_file(f"{base}/{vid}.mp4", vp)
        if au is None:
            if vp.exists() and not ap.exists():
                extract_wav_from_video(vp, ap)
            au = mel_from_wav_path(ap, mx)
            ra = True
        if vi is None:
            vv = _frames(vp, mx)
            vi = F.interpolate(vv, (14, 14), mode="bilinear").mean(1, True).repeat(1, 512, 1, 1)
            rv = True
    if strict:
        if rv and not vp.exists():
            raise FileNotFoundError(vp)
        if ra and not ap.exists() and vp.exists():
            raise FileNotFoundError(ap)
    q = torch.tensor([w2i.get(t, 0) for t in _tokenize(item)] or [0], dtype=torch.long)
    ans = item.get("answer", item.get("anser", "unknown"))
    if ans not in a2i:
        li, lb = -1, torch.tensor([-1], dtype=torch.long)
    else:
        li = a2i[ans]
        lb = torch.tensor([li], dtype=torch.long)
    stt = {"video": vi, "audio": au, "question": q, "label": lb.view(1)}
    meta = {
        "question_id": item.get("question_id", idx),
        "question_type": item.get("type", "[]"),
        "video_id": vid,
    }
    return stt, li, meta


def _ann_path(root, split, prepare, dcfg):
    root = Path(root).resolve()
    ap = root / "annotations" / f"{split}.json"
    if prepare:
        root.mkdir(parents=True, exist_ok=True)
        ap.parent.mkdir(parents=True, exist_ok=True)
        (root / "video").mkdir(parents=True, exist_ok=True)
        (root / "audio").mkdir(parents=True, exist_ok=True)
        u = (dcfg.get("annotations") or {}).get(split)
        if u and not ap.exists():
            download_file(str(u), ap)
    if not ap.exists():
        raise RuntimeError(str(ap))
    return ap


def _materialize(root, split, data, w2i, a2i, mx, st, strict, uf, dcfg, feats):
    root = Path(root).resolve()
    fad = Path(feats.get("audio_dir", root / "feats" / "vggish")).resolve()
    fvd = Path(feats.get("visual_dir", root / "feats" / "res18_14x14")).resolve()
    outd = root / "cache" / split
    outd.mkdir(parents=True, exist_ok=True)
    idx = []
    for i, row in enumerate(tqdm(data, desc=split)):
        tens, li, meta = _pack(row, i, root, mx, st, strict, uf, dcfg, fad, fvd, w2i, a2i)
        p = outd / f"{i:08d}.safetensors"
        safetensors.torch.save_file({k: v.contiguous() for k, v in tens.items()}, str(p))
        idx.append({"path": str(p), "label": li, **meta})
    write_json(outd / "index.json", idx)
    return idx


class MusicAVQADataset(BaseDataset):
    def __init__(
        self,
        root_dir,
        split="train",
        max_len=16,
        prepare_data=True,
        download=None,
        strict_media=False,
        max_samples=None,
        answer_to_idx=None,
        word_to_idx=None,
        use_official_features=True,
        features=None,
        frame_stride=6,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
        **kwargs,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.split = split
        dl = download or {}
        feats = features or {}
        ix_path = self.root_dir / "cache" / split / "index.json"
        voc_path = self.root_dir / "cache" / "vocabs.json"
        if not ix_path.exists():
            ap = _ann_path(self.root_dir, split, prepare_data, dl)
            with ap.open("r", encoding="utf-8") as f:
                rows = json.load(f)
            if max_samples:
                rows = rows[: int(max_samples)]
            if split == "train" and (answer_to_idx is None or word_to_idx is None):
                word_to_idx, answer_to_idx = _vocabs(rows)
            elif word_to_idx is None or answer_to_idx is None:
                raise ValueError("word_to_idx and answer_to_idx required for this split")
            index = _materialize(
                self.root_dir, split, rows, word_to_idx, answer_to_idx,
                max_len, frame_stride, strict_media, use_official_features, dl, feats,
            )
            if split == "train":
                write_json(voc_path, {"word_to_idx": word_to_idx, "answer_to_idx": answer_to_idx})
        else:
            index = read_json(ix_path)
            if split == "train":
                v = read_json(voc_path)
                word_to_idx, answer_to_idx = v["word_to_idx"], v["answer_to_idx"]
            elif word_to_idx is None or answer_to_idx is None:
                raise ValueError("word_to_idx and answer_to_idx required")
        self.word_to_idx = word_to_idx
        self.answer_to_idx = answer_to_idx
        self.num_classes = len(answer_to_idx)
        super().__init__(index, limit, shuffle_index, instance_transforms)

    @property
    def answer_vocab(self):
        return self.answer_to_idx

    def __getitem__(self, ind):
        e = self._index[ind]
        t = safetensors.torch.load_file(e["path"])
        out = {
            "video": t["video"],
            "audio": t["audio"],
            "question": t["question"],
            "label": t["label"].reshape(-1)[0].long(),
            "question_id": e["question_id"],
            "question_type": e["question_type"],
            "video_id": e["video_id"],
        }
        if self.instance_transforms:
            for k, fn in self.instance_transforms.items():
                if k in out and torch.is_tensor(out[k]):
                    out[k] = fn(out[k])
        return out

    @staticmethod
    def collate_batch(batch):
        labels = torch.stack([s["label"] for s in batch]).long()
        video = torch.stack([s["video"] for s in batch])
        audio = torch.stack([s["audio"] for s in batch])
        qs = [s["question"] for s in batch]
        m = max(q.size(0) for q in qs)
        pad = []
        for q in qs:
            pad.append(torch.cat([q, torch.zeros(m - q.size(0), dtype=torch.long)]))
        return {
            "video": video,
            "audio": audio,
            "question": torch.stack(pad),
            "label": labels,
            "question_id": [s.get("question_id", -1) for s in batch],
            "question_type": [s.get("question_type", "[]") for s in batch],
            "video_id": [s.get("video_id", "") for s in batch],
        }
