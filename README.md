# Tacotron2
Pytorch implementation of Tacotron2, a modern text-to-speech model based on [this paper](https://arxiv.org/pdf/1712.05884v2.pdf)

## Usage
To convert mel spectrograms to audio we need Nvidia's pretrained Vocoder

```python
! git clone https://github.com/NVIDIA/waveglow.git

! pip install googledrivedownloader

from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(
    file_id='1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF',
    dest_path='./waveglow_256channels_universal_v5.pt'
)
```

Then run `./run_docker.sh` with correct `volume` option

### Training

Download [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/)

Set preferred settings in `config.py`, then run
`python train.py`

In `wandb.ai` will be logged:

- Train and validation losses
- Original text
- Predicted and ground truth mel spectrograms
- Predicted and ground truth audio
- Probabilties of the last frame over the audio
    

### Inference

`python inference.py "Your text for speech synthesis" `

The result will be saved in `audio.wav`.
