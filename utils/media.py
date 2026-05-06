import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio


def trim_pad_time(x, L, stride=1):
    if stride > 1:
        x = x[::stride]
    n = x.size(0)
    if n > L:
        return x[:L]
    if n < L:
        z = torch.zeros((L - n,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
        return torch.cat([x, z], 0)
    return x


def mel_from_wav_path(path, L, n_mels=128, n_fft=1024, hop=256):
    w, sr = mono_wav_tensor(path)
    m = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop
    )(w).squeeze(0).T
    n = m.size(0)
    if n > L:
        return m[:L]
    if n < L:
        return torch.cat([m, torch.zeros(L - n, m.size(1), device=m.device)], 0)
    return m


def extract_wav_from_video(video_path, wav_path):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg required for audio extraction")
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = wav_path.with_suffix(".tmp.wav")
    try:
        subprocess.run(
            [
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
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError((e.stderr or "") + (e.stdout or "")) from e
    tmp.replace(wav_path)


def mono_wav_tensor(path):
    path = Path(path)
    if not path.exists():
        return torch.zeros(1, 1), 16000
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32).reshape(-1, nch).mean(1) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32).reshape(-1, nch).mean(1) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32).reshape(-1, nch).mean(1) - 128.0) / 128.0
    else:
        raise ValueError(sw)
    return torch.from_numpy(x).float().unsqueeze(0), sr
