import torch
from torch.utils.data import Dataset

import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.io as io
import torchaudio
import subprocess
from tqdm import tqdm


class MusicAVQADataset(Dataset):
    def __init__(
        self,
        root_dir="./data/music_avqa",
        split="train",
        max_len=32,
        download=True
    ):
        self.root_dir = root_dir
        self.split = split
        self.max_len = max_len

        self.video_dir = os.path.join(root_dir, "videos")
        self.audio_dir = os.path.join(root_dir, "audio")
        self.ann_path = os.path.join(root_dir, f"{split}.json")

        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

        if download:
            self._download_annotations()
            self._download_media()

        with open(self.ann_path) as f:
            self.annotations = json.load(f)

    def _download_annotations(self):
        if os.path.exists(self.ann_path):
            return

        url = "https://raw.githubusercontent.com/xiaobai1217/MUSIC-AVQA/master/dataset/json/train.json"

        subprocess.run([
            "wget", "-O", self.ann_path, url
        ])

    def _download_media(self):
        with open(self.ann_path) as f:
            data = json.load(f)

        video_ids = set(item["video_id"] for item in data)

        for vid in tqdm(video_ids):
            video_path = os.path.join(self.video_dir, f"{vid}.mp4")
            audio_path = os.path.join(self.audio_dir, f"{vid}.wav")

            if not os.path.exists(video_path):
                url = f"https://www.youtube.com/watch?v={vid}"
                try:
                    subprocess.run([
                        "yt-dlp",
                        url,
                        "-o", video_path,
                        "-f", "mp4"
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except:
                    continue

            if os.path.exists(video_path) and not os.path.exists(audio_path):
                subprocess.run([
                    "ffmpeg",
                    "-i", video_path,
                    "-q:a", "0",
                    "-map", "a",
                    audio_path,
                    "-y"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    def load_video(self, path):
        video, _, _ = io.read_video(path, pts_unit='sec')  # (T, H, W, C)
        video = video.permute(0, 3, 1, 2)  # (T, C, H, W)

        if video.shape[0] > self.max_len:
            video = video[:self.max_len]
        else:
            pad = self.max_len - video.shape[0]
            video = torch.cat([video, torch.zeros(pad, *video.shape[1:])])

        return video

    def load_audio(self, path):
        audio, sr = torchaudio.load(path)

        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(audio)
        mel = mel.squeeze(0).transpose(0, 1)

        if mel.shape[0] > self.max_len:
            mel = mel[:self.max_len]
        else:
            pad = self.max_len - mel.shape[0]
            mel = torch.cat([mel, torch.zeros(pad, mel.shape[1])])

        return mel

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]

        vid = item["video_id"]

        video_path = os.path.join(self.video_dir, f"{vid}.mp4")
        audio_path = os.path.join(self.audio_dir, f"{vid}.wav")

        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            return self.__getitem__((idx + 1) % len(self))

        video = self.load_video(video_path)
        audio = self.load_audio(audio_path)

        label = torch.tensor(item.get("answer", 0)).long()

        return video, audio, label