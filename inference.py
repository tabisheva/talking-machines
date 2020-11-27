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


df = pd.read_csv("LJSpeech-1.1/metadata.csv", sep='|', quotechar='`', index_col=0, header=None)
train, test = train_test_split(df, test_size=0.2, random_state=10)

train_dataset = LJDataset(train)
test_dataset = LJDataset(test)

model_config = ModelConfig()

train_dataloader = DataLoader(train_dataset,
                              batch_size=model_config.batch_size,
                              num_workers=model_config.num_workers,
                              shuffle=False,
                              collate_fn=collate_fn,
                              pin_memory=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=model_config.num_workers,
                             collate_fn=collate_fn,
                             pin_memory=True)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model_config = ModelConfig()
model = Tacotron2(model_config, device)
model.load_state_dict(torch.load(model_config.model_path, map_location=device))
model.to(device)
vocoder = Vocoder(device)
vocoder.eval()
if model_config.wandb_log:
    wandb.init(project="text2speech")
    wandb.watch(model, log="all")

model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        x, y = model.parse_batch(batch)
        y_pred = model.inference(x)
        break

test_wav = vocoder.inference(y_pred[0][:1]).squeeze()
original_text = "".join([idxToChar[c] for c in x[0][0].detach().cpu().numpy()])

if model_config.wandb_log:
    wandb.log({
               "inference audio": [wandb.Audio(test_wav.cpu(), caption=original_text, sample_rate=22050)],
               })
    for prob in y_pred[2][0].squeeze().detach().cpu().numpy():
        wandb.log({
               "probs stop token": prob,
               })
