import torch
import torch.nn as nn
from src.preprocessing import get_mask_from_lengths

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets, input_lengths):
        mel_target, gate_target = targets[0], targets[1]
        mel_out, mel_out_postnet, gate_out, _ = model_output
        # (B, T)
        mask_gate_loss = get_mask_from_lengths(input_lengths, mel_out.device)
#        mask_gate_loss = torch.ones_like(gate_out).masked_fill(mask, 0.0)
        gate_loss = torch.sum(mask_gate_loss * nn.BCEWithLogitsLoss(reduction='none')(gate_out, gate_target)) / \
                    mask_gate_loss.sum().item()
        # get (B, 80, T) mask from (B, T)
        mask_mel_loss = mask_gate_loss.expand(80, mask_gate_loss.size(0), mask_gate_loss.size(1)).permute(1, 0, 2)

        mel_loss_before_postnet = torch.sum(mask_mel_loss * nn.MSELoss(reduction='none')(mel_out, mel_target)) / \
                                  mask_mel_loss.sum().item()
        mel_loss_after_postnet = torch.sum(mask_mel_loss * nn.MSELoss(reduction='none')(mel_out_postnet, mel_target)) / \
                                 mask_mel_loss.sum().item()
        return mel_loss_before_postnet + mel_loss_after_postnet + gate_loss
