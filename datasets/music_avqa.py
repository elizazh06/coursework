import ast
import json
import shutil
import subprocess
import urllib.request
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio
from decord import VideoReader, cpu
from torch.utils.data import Dataset


def _download_file(url: str, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[MusicAVQA] Downloading {url} -> {destination}")
    req = urllib.request.Request(url, headers={"User-Agent": "MusicAVQA-dataset/1.0"})
    with urllib.request.urlopen(req) as resp, open(destination, "wb") as out:
        shutil.copyfileobj(resp, out)


def _extract_wav_from_video(video_path: Path, wav_path: Path) -> None:
    """Cache mono 16 kHz PCM WAV extracted from video (ffmpeg)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required to extract audio from video. "
            "Install ffmpeg (e.g. apt install ffmpeg) or use an image that includes it."
        )
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = wav_path.with_suffix(".tmp.wav")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "") + (e.stdout or "")
        raise RuntimeError(f"ffmpeg failed for {video_path}: {err}") from e
    tmp.replace(wav_path)


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
        self.root_dir = Path(root_dir).resolve()
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
        ).resolve()
        self.feature_visual_dir = Path(
            self.features_cfg.get("visual_dir", self.root_dir / "feats" / "res18_14x14")
        ).resolve()
        self.ann_path = self.annotations_dir / f"{split}.json"

        if prepare_data:
            self.prepare_data()

        if not self.ann_path.exists():
            raise RuntimeError(
                f"{self.ann_path} not found. "
                "Place JSON annotations under data/music_avqa/annotations/ "
                "or set download.annotations.<split> to a URL in config."
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
                self._download_file(str(ann_url), self.ann_path)

    def _download_file(self, url, destination):
        _download_file(str(url), Path(destination))

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

    def _ensure_video(self, video_id: str, video_path: Path) -> None:
        media_cfg = self.download_cfg.get("media", {})
        video_base_url = str(media_cfg.get("video_base_url", "")).rstrip("/")

        if video_path.exists():
            return
        if not video_base_url:
            if self.strict_media:
                raise FileNotFoundError(
                    f"Missing video {video_path} and download.media.video_base_url is not set."
                )
            return
        url = f"{video_base_url}/{video_id}.mp4"
        self._download_file(url, video_path)

    def _ensure_audio_wav(self, video_id: str, video_path: Path, audio_path: Path) -> None:
        """Derive cached WAV from video when raw audio features are not used."""
        if audio_path.exists():
            return
        if not video_path.exists():
            if self.strict_media:
                raise FileNotFoundError(f"Cannot extract audio: missing video {video_path}")
            return
        _extract_wav_from_video(video_path, audio_path)

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

    def _load_wav_pcm(self, audio_path: Path) -> tuple[torch.Tensor, int]:
        """Load mono/stereo PCM WAV without torchaudio.load (avoids torchcodec dependency)."""
        with wave.open(str(audio_path), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
        if sw == 2:
            flat = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            x = flat.reshape(-1, nch).mean(axis=1)
        elif sw == 4:
            flat = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            x = flat.reshape(-1, nch).mean(axis=1)
        elif sw == 1:
            flat = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            x = (flat.reshape(-1, nch).mean(axis=1) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported WAV sample width {sw} in {audio_path}")
        waveform = torch.from_numpy(x).unsqueeze(0)
        return waveform, sr

    def _load_audio(self, audio_path):
        if not Path(audio_path).exists():
            return torch.zeros(self.max_len, 128)

        waveform, sr = self._load_wav_pcm(Path(audio_path))
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=128,
            n_fft=1024,
            hop_length=256,
        )(waveform)

        mel = mel.squeeze(0).transpose(0, 1)

        if mel.size(0) > self.max_len:
            mel = mel[: self.max_len]
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
        used_raw_video = False
        used_raw_audio = False

        if self.use_official_features:
            audio = self._load_feature_audio(video_id)
            visual_posi = self._load_feature_visual(video_id)

        video_path = self.video_dir / f"{video_id}.mp4"
        audio_path = self.audio_dir / f"{video_id}.wav"

        if audio is None or visual_posi is None:
            self._ensure_video(video_id, video_path)

            if audio is None:
                self._ensure_audio_wav(video_id, video_path, audio_path)
                audio = self._load_audio(audio_path)
                used_raw_audio = True

            if visual_posi is None:
                video = self._load_video(video_path)
                visual_posi = torch.nn.functional.interpolate(video, size=(14, 14), mode="bilinear")
                visual_posi = visual_posi.mean(dim=1, keepdim=True).repeat(1, 512, 1, 1)
                used_raw_video = True

        if self.strict_media:
            if used_raw_video and not video_path.exists():
                raise FileNotFoundError(
                    f"Missing video for {video_id}: {video_path}. "
                    "Check download.media.video_base_url or set strict_media=false."
                )
            if used_raw_audio and not audio_path.exists():
                raise FileNotFoundError(
                    f"Missing extracted audio for {video_id}: {audio_path}. "
                    "Ensure ffmpeg can read the video and write to audio cache."
                )

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
