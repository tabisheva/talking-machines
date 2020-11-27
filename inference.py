import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from src.dataset import LJDataset
from src.model import Tacotron2
from src.loss_function import Tacotron2Loss
from src.preprocessing import collate_fn
from src.dataset import idxToChar
from src.vocoder import Vocoder
from config import ModelConfig
import warnings
warnings.filterwarnings('ignore')
import sys

vocab = " abcdefghijklmnopqrstuvwxyz.,:;-?!"
charToIdx = {c: i for i, c in enumerate(vocab)}

model_config = ModelConfig()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model_config = ModelConfig()
model = Tacotron2(model_config, device)
model.load_state_dict(torch.load(model_config.model_path, map_location=device))
model.to(device)
vocoder = Vocoder(device)
vocoder.eval()

text = sys.argv[1].lower().strip()
text_ids = torch.LongTensor(np.array([charToIdx[c] for c in text if c in vocab])).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    y_pred = model.inference(torch.tensor(text_ids))

test_wav = vocoder.inference(y_pred[0][:1]).squeeze()
wandb.init(project="text2speech")
wandb.log({
               "inference audio": [wandb.Audio(test_wav.cpu(), caption=text, sample_rate=22050)],
               })
for prob in y_pred[2][0].squeeze().detach().cpu().numpy():
    wandb.log({
               "probs stop token": prob,
               })
