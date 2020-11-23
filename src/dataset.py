import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchaudio
from src.preprocessing import MelSpectrogram
from config import MelSpectrogramConfig

vocab = " abcdefghijklmnopqrstuvwxyz.,:;-?!"

charToIdx = {c: i for i, c in enumerate(vocab)}
idxToChar = {i: c for i, c in enumerate(vocab)}


class LJDataset(Dataset):
    def __init__(self, df):
        self.dir = "LJSpeech-1.1/wavs"
        self.filenames = df.index.values
        self.labels = df[2].values
        self.featurizer = MelSpectrogram(MelSpectrogramConfig())

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        text = self.labels[idx].lower().strip()
        text = torch.IntTensor(np.array([charToIdx[c] for c in text if c in vocab]))
        wav, sr = torchaudio.load(os.path.join(self.dir, f'{filename}.wav'))
        wav = wav.squeeze()
        mel = self.featurizer(wav)
        return text, mel.T  # (time, num_channels)

    def __len__(self):
        return len(self.filenames)
