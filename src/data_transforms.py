import torch
import torch.nn as nn
import torchaudio
import librosa
from config import MelSpectrogramConfig


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels
        )

        self.mel_spectrogram.spectrogram.power = config.power

        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel


def collate_fn(batch):
    """
    Stacking sequences of variable lengths in batches
    :param batch: list of tuples with texts (num_chars, ) and mels (num_channels, T)
    :return: tensor (batch, max_length of texts) with zero-padded texts,
             tensor (batch, ) with texts lengths,
             tensor (batch, num_channels, max_time) with padded mels,
             tensor (batch, max_time) with padded gates
             tensor (batch, ) with mel lengths
    """
    texts, mels = list(zip(*batch))
    texts_length = torch.LongTensor([len(text) for text in texts])
    mels_length = torch.LongTensor([mel.shape[0] for mel in mels])
    # (B, max_text_length)
    texts_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
    # (B, num_channels, max_time)
    mels_padded = torch.nn.utils.rnn.pad_sequence(mels,
                                                  batch_first=True,
                                                  padding_value=MelSpectrogramConfig.pad_value).permute(0, 2, 1)
    # tensor with shape (B, max_time) for binary classification of stop token
    # we put zeros for all frames except the last one
    max_mel_len = max(mels_length)
    ids = torch.arange(max_mel_len)
    gates_padded = (~(ids < mels_length.unsqueeze(1) - 1)).long()
    # input_lengths, ids_sorted_decreasing = torch.sort(
    #     torch.LongTensor([len(x[0]) for x in batch]),
    #     dim=0, descending=True)
    # max_input_len = input_lengths[0]
    # text_padded = torch.LongTensor(len(batch), max_input_len)
    # text_padded.zero_()
    # for i in range(len(ids_sorted_decreasing)):
    #     text = batch[ids_sorted_decreasing[i]][0]
    #     text_padded[i, :text.size(0)] = text
    # num_mels = batch[0][1].size(0)
    # max_target_len = max([x[1].size(1) for x in batch])
    #
    # # include mel padded and gate padded
    # mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
    # mel_padded.fill_(MelSpectrogramConfig.pad_value)
    # gate_padded = torch.FloatTensor(len(batch), max_target_len)
    # gate_padded.zero_()
    # output_lengths = torch.LongTensor(len(batch))
    # for i in range(len(ids_sorted_decreasing)):
    #     mel = batch[ids_sorted_decreasing[i]][1]
    #     mel_padded[i, :, :mel.size(1)] = mel
    #     gate_padded[i, mel.size(1) - 1:] = 1
    #     output_lengths[i] = mel.size(1)
    #    return text_padded, input_lengths, mel_padded, gate_padded, output_lengths

    return texts_padded, texts_length, mels_padded, gates_padded, mels_length
