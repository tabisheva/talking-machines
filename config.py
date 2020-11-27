class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    pad_value: float = -11.5129251


class ModelConfig:
    input_dim: int = 34
    batch_size: int = 20
    num_workers: int = 8
    from_pretrained: bool = False
    lr: float = 0.001
    num_epochs: int = 20
    start_epoch: int = 3
    model_path: str = "tacotron_3.pth"
    wandb_log: bool = True
    num_channels: int = 80


class PostnetConfig:
    params = [
        {"in_channels": 80, "out_channels": 512},
        {"in_channels": 512, "out_channels": 512},
        {"in_channels": 512, "out_channels": 512},
        {"in_channels": 512, "out_channels": 512},
        {"in_channels": 512, "out_channels": 80},
    ]
