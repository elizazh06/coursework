import ast
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
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
        answer_to_idx=None,
        word_to_idx=None,
        use_official_features=True,
        features=None,
        frame_stride=6,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.strict_media = strict_media
        self.download_cfg = download or {}
        self.use_official_features = use_official_features
        self.frame_stride = frame_stride
        self.features_cfg = features or {}

        self.annotations_dir = self.root_dir / "annotations"
        self.video_dir = self.root_dir / "video"
        self.audio_dir = self.root_dir / "audio"
        self.feature_audio_dir = Path(
            self.features_cfg.get("audio_dir", self.root_dir / "feats" / "vggish")
        )
        self.feature_visual_dir = Path(
            self.features_cfg.get("visual_dir", self.root_dir / "feats" / "res18_14x14")
        )
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

        self.word_to_idx = word_to_idx if word_to_idx is not None else self._build_word_vocab(self.data)
        self.answer_to_idx = (
            answer_to_idx if answer_to_idx is not None else self._build_answer_vocab(self.data)
        )
        self.num_classes = len(self.answer_vocab)

    @property
    def answer_vocab(self):
        return self.answer_to_idx

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

    def _build_answer_vocab(self, items):
        answer_to_idx = {}
        for item in items:
            answer = item.get("answer", item.get("anser", "unknown"))
            if answer not in answer_to_idx:
                answer_to_idx[answer] = len(answer_to_idx)
        return answer_to_idx

    def _tokenize_question(self, item):
        question = item["question_content"].rstrip().split(" ")
        if question and question[-1].endswith("?"):
            question[-1] = question[-1][:-1]

        templ_values = item.get("templ_values", "[]")
        try:
            templ_values = ast.literal_eval(templ_values) if isinstance(templ_values, str) else templ_values
        except (ValueError, SyntaxError):
            templ_values = []

        pointer = 0
        for i, token in enumerate(question):
            if "<" in token and pointer < len(templ_values):
                question[i] = str(templ_values[pointer])
                pointer += 1

        return [w.lower() for w in question if w]

    def _build_word_vocab(self, items):
        vocab = {" ": 0}
        for item in items:
            for token in self._tokenize_question(item):
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab

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

    def _time_sample_and_pad(self, tensor_2d, max_len):
        sampled = tensor_2d[:: self.frame_stride]
        if sampled.size(0) > max_len:
            sampled = sampled[:max_len]
        elif sampled.size(0) < max_len:
            pad = max_len - sampled.size(0)
            sampled = torch.cat([sampled, torch.zeros(pad, sampled.size(1), dtype=sampled.dtype)], dim=0)
        return sampled

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

    def _load_feature_audio(self, video_id):
        feature_path = self.feature_audio_dir / f"{video_id}.npy"
        if not feature_path.exists():
            return None
        audio = torch.from_numpy(np.load(feature_path)).float()
        return self._time_sample_and_pad(audio, self.max_len)

    def _load_feature_visual(self, video_id):
        feature_path = self.feature_visual_dir / f"{video_id}.npy"
        if not feature_path.exists():
            return None
        visual = torch.from_numpy(np.load(feature_path)).float()
        visual = visual[:: self.frame_stride]
        if visual.size(0) > self.max_len:
            visual = visual[: self.max_len]
        elif visual.size(0) < self.max_len:
            pad_t = self.max_len - visual.size(0)
            pad = torch.zeros(pad_t, visual.size(1), visual.size(2), visual.size(3), dtype=visual.dtype)
            visual = torch.cat([visual, pad], dim=0)
        return visual

    def _encode_question(self, question):
        encoded = [self.word_to_idx.get(token, 0) for token in question]
        if not encoded:
            encoded = [0]
        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.data[idx]

        video_id = item["video_id"]
        question = self._tokenize_question(item)
        answer = item.get("answer", item.get("anser", "unknown"))
        question_id = item.get("question_id", idx)
        question_type = item.get("type", "[]")

        visual_posi = None
        audio = None

        if self.use_official_features:
            audio = self._load_feature_audio(video_id)
            visual_posi = self._load_feature_visual(video_id)

        if audio is None or visual_posi is None:
            video_path = self.video_dir / f"{video_id}.mp4"
            audio_path = self.audio_dir / f"{video_id}.wav"
            self._maybe_download_media(video_id, video_path, audio_path)

            if audio is None:
                audio = self._load_audio(audio_path)

            if visual_posi is None:
                video = self._load_video(video_path)
                # Raw-video fallback: convert to pseudo [T, 512, 14, 14] map.
                visual_posi = torch.nn.functional.interpolate(video, size=(14, 14), mode="bilinear")
                visual_posi = visual_posi.mean(dim=1, keepdim=True).repeat(1, 512, 1, 1)

        question = self._encode_question(question)

        if answer not in self.answer_to_idx:
            label = torch.tensor(-1, dtype=torch.long)
        else:
            label = torch.tensor(self.answer_to_idx[answer], dtype=torch.long)

        return {
            "video": visual_posi,
            "audio": audio,
            "question": question,
            "label": label,
            "question_id": question_id,
            "question_type": question_type,
            "video_id": video_id,
        }