import os
import json
import torch
from torch.utils.data import Dataset
import torchaudio
from decord import VideoReader, cpu


class MusicAVQADataset(Dataset):
    def __init__(self, root_dir, split="train", max_len=16):
        self.root_dir = root_dir
        self.split = split
        self.max_len = max_len

        ann_path = os.path.join(root_dir, "annotations", f"{split}.json")

        if not os.path.exists(ann_path):
            raise RuntimeError(
                f"{ann_path} not found. Download annotations manually."
            )

        with open(ann_path, "r") as f:
            self.data = json.load(f)
        
        self.answer_vocab = {}
        self.answer_to_idx = {}

        for item in self.data:
            ans = item["anser"]
            if ans not in self.answer_vocab:
                self.answer_vocab[ans] = len(self.answer_vocab)

        self.answer_to_idx = self.answer_vocab
        self.num_classes = len(self.answer_vocab)

    def __len__(self):
        return len(self.data)

    def _load_video(self, video_path):
        if not os.path.exists(video_path):
            return torch.randn(self.max_len, 3, 224, 224)

        vr = VideoReader(video_path, ctx=cpu(0))
        frames = vr[:self.max_len]

        video = torch.tensor(frames).permute(0, 3, 1, 2)
        return video

    def _load_audio(self, audio_path):
        if not os.path.exists(audio_path):
            return torch.randn(self.max_len, 128)

        waveform, sr = torchaudio.load(audio_path)
        mel = torchaudio.transforms.MelSpectrogram(sr)(waveform)

        mel = mel.squeeze(0).transpose(0, 1)

        if mel.size(0) > self.max_len:
            mel = mel[:self.max_len]

        return mel

    def _encode_question(self, question):
        tokens = question.lower().split()
        return torch.tensor([hash(w) % 5000 for w in tokens], dtype=torch.long)

    def __getitem__(self, idx):
        item = self.data[idx]

        video_id = item["video_id"]
        question = item["question_content"]
        answer = item["anser"]

        video_path = os.path.join(self.root_dir, "video", f"{video_id}.mp4")
        audio_path = os.path.join(self.root_dir, "audio", f"{video_id}.wav")

        video = self._load_video(video_path)
        audio = self._load_audio(audio_path)
        question = self._encode_question(question)

        label = torch.tensor(self.answer_to_idx[answer], dtype=torch.long)

        return video, audio, question, label