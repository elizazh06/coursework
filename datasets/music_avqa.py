import json
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import torch
import torchaudio
from decord import VideoReader, cpu
from torch.utils.data import Dataset


class MusicAVQADataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        max_len=16,
        vocab_size=5000,
        prepare_data=True,
        download=None,
        strict_media=False,
        max_samples=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.strict_media = strict_media
        self.download_cfg = download or {}

        self.annotations_dir = self.root_dir / "annotations"
        self.video_dir = self.root_dir / "video"
        self.audio_dir = self.root_dir / "audio"
        self.ann_path = self.annotations_dir / f"{split}.json"

        if prepare_data:
            self.prepare_data()

        if not self.ann_path.exists():
            raise RuntimeError(
                f"{self.ann_path} not found. "
                "Provide download.annotations.<split> URL in config or prepare data manually."
            )

        with self.ann_path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)
        if max_samples is not None:
            self.data = self.data[: int(max_samples)]

        self.answer_vocab = {}
        for item in self.data:
            ans = item.get("answer", item.get("anser", "unknown"))
            if ans not in self.answer_vocab:
                self.answer_vocab[ans] = len(self.answer_vocab)

        self.answer_to_idx = self.answer_vocab
        self.num_classes = len(self.answer_vocab)

    def prepare_data(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        if not self.ann_path.exists():
            ann_url = self.download_cfg.get("annotations", {}).get(self.split)
            if ann_url:
                self._download_file(ann_url, self.ann_path)

        archives = self.download_cfg.get("archives", [])
        for archive in archives:
            url = archive.get("url")
            if not url:
                continue
            destination = self.root_dir / archive.get("filename", Path(url).name)
            extract_to = self.root_dir / archive.get("extract_to", "")
            if not destination.exists():
                self._download_file(url, destination)
            self._extract_archive(destination, extract_to)

    def _download_file(self, url, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[MusicAVQA] Downloading {url} -> {destination}")
        urllib.request.urlretrieve(url, destination)

    def _extract_archive(self, archive_path, target_dir):
        archive_path = Path(archive_path)
        target_dir = Path(target_dir)
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(target_dir)
        elif archive_path.suffix in {".tgz", ".gz"} or archive_path.name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(target_dir)
        elif archive_path.suffix == ".tar":
            with tarfile.open(archive_path, "r") as tf:
                tf.extractall(target_dir)

    def _maybe_download_media(self, video_id, video_path, audio_path):
        media_cfg = self.download_cfg.get("media", {})
        video_base_url = media_cfg.get("video_base_url", "").rstrip("/")
        audio_base_url = media_cfg.get("audio_base_url", "").rstrip("/")

        if not video_path.exists() and video_base_url:
            self._download_file(f"{video_base_url}/{video_id}.mp4", video_path)

        if not audio_path.exists() and audio_base_url:
            self._download_file(f"{audio_base_url}/{video_id}.wav", audio_path)

        if self.strict_media and (not video_path.exists() or not audio_path.exists()):
            raise FileNotFoundError(
                f"Missing media for {video_id}. "
                "Provide media base URLs in config.download.media or set strict_media=false."
            )

    def __len__(self):
        return len(self.data)

    def _load_video(self, video_path):
        if not Path(video_path).exists():
            return torch.zeros(self.max_len, 3, 224, 224)

        vr = VideoReader(str(video_path), ctx=cpu(0))
        frame_count = len(vr)
        if frame_count == 0:
            return torch.zeros(self.max_len, 3, 224, 224)

        indices = torch.linspace(0, frame_count - 1, steps=self.max_len).long().tolist()
        frames = vr.get_batch(indices)

        video = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float() / 255.0
        return video

    def _load_audio(self, audio_path):
        if not Path(audio_path).exists():
            return torch.zeros(self.max_len, 128)

        waveform, sr = torchaudio.load(audio_path)
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=128)(waveform)

        mel = mel.squeeze(0).transpose(0, 1)

        if mel.size(0) > self.max_len:
            mel = mel[:self.max_len]
        elif mel.size(0) < self.max_len:
            pad = self.max_len - mel.size(0)
            mel = torch.cat([mel, torch.zeros(pad, mel.size(1))], dim=0)

        return mel

    def _encode_question(self, question):
        tokens = question.lower().split()
        hashed = [hash(w) % self.vocab_size for w in tokens]
        if not hashed:
            hashed = [0]
        return torch.tensor(hashed, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.data[idx]

        video_id = item["video_id"]
        question = item["question_content"]
        answer = item.get("answer", item.get("anser", "unknown"))

        video_path = self.video_dir / f"{video_id}.mp4"
        audio_path = self.audio_dir / f"{video_id}.wav"
        self._maybe_download_media(video_id, video_path, audio_path)

        video = self._load_video(video_path)
        audio = self._load_audio(audio_path)
        question = self._encode_question(question)

        label = torch.tensor(self.answer_to_idx[answer], dtype=torch.long)

        return {
            "video": video,
            "audio": audio,
            "question": question,
            "label": label,
        }