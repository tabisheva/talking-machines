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
                             batch_size=model_config.batch_size,
                             num_workers=model_config.num_workers,
                             collate_fn=collate_fn,
                             pin_memory=True)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = Tacotron2(model_config, device)
vocoder = Vocoder(device)
vocoder.eval()
if model_config.from_pretrained:
    model.load_state_dict(torch.load(model_config.model_path))
model.to(device)
criterion = Tacotron2Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.lr)
# num_steps = len(train_dataloader) * model_config.num_epochs
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.00001)

if model_config.wandb_log:
    wandb.init(project="text2speech")
    wandb.watch(model, log="all")

best_loss = 10.0
start_epoch = model_config.start_epoch + 1 if model_config.from_pretrained else 1
for epoch in range(start_epoch, model_config.num_epochs + 1):
    model.to(device)
    train_losses = []
    model.train()
    for batch in train_dataloader:
        x, y = model.parse_batch(batch)
        y_pred = model(x)
        loss = criterion(y_pred, y, x[3])  # inputs lengths for masked loss
        optimizer.zero_grad()
        loss.backward()
        print(epoch, loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        # lr_scheduler.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_losses = []
        for batch in test_dataloader:
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y, x[3])
            val_losses.append(loss.item())
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    test_wav = vocoder.inference(y_pred[0][:1]).squeeze()
    original_text = "".join([idxToChar[c] for c in x[0][0].detach().cpu().numpy()])

    if model_config.wandb_log:
        wandb.log({"train_loss": train_loss,
                   "val_loss": val_loss,
                   "test audio teacher forcing": [wandb.Audio(test_wav.cpu(), caption=original_text, sample_rate=22050)],
                   "predicted melspec": [wandb.Image(y_pred[1][0].detach().cpu().numpy())],
                   "attention": [wandb.Image(y_pred[3][0].T.detach().cpu().numpy())],
                   "ground truth melspec": [wandb.Image(y[0][0].detach().cpu().numpy())]
                   })

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "tacotron_mon.pth")
