import torch
import torch.nn as nn
from src.preprocessing import get_mask_from_lengths


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets, input_lengths):
        mel_target, gate_target = targets[0], targets[1]
        mel_out, mel_out_postnet, gate_out, attn_out = model_output
        # get 2d mask from input_lengths (B, T)
        mask_gate_loss = get_mask_from_lengths(input_lengths, mel_out.device)
        # guided attention
        _, decoder_steps, num_chars = attn_out.shape
        grid_t, grid_n = torch.meshgrid(torch.arange(decoder_steps, device=attn_out.device),
                                        torch.arange(num_chars, device=attn_out.device))
        attn_mask = 1 - torch.exp(-(-grid_n / num_chars + grid_t / decoder_steps) ** 2 / 0.08)
        # guided attention loss with mask

        attn_loss = attn_out * attn_mask[None] * mask_gate_loss.expand(attn_out.shape[-1], mask_gate_loss.size(0),
                                                                       mask_gate_loss.size(1)).permute(1, 2, 0)
        # normalized attention loss
        attn_loss = torch.sum(attn_loss) / mask_gate_loss.sum().item()
        # gate loss with large positive weights because of imbalanced classes
        gate_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([1000.0]).to(gate_out.device))(
            gate_out, gate_target)
        # normalized gate loss
        gate_loss = torch.sum(mask_gate_loss * gate_loss) / mask_gate_loss.sum().item()
        # get (B, 80, T) mask from (B, T) for mel specs
        mask_mel_loss = mask_gate_loss.expand(80, mask_gate_loss.size(0), mask_gate_loss.size(1)).permute(1, 0, 2)

        mel_loss_before_postnet = mask_mel_loss * nn.MSELoss(reduction='none')(mel_out, mel_target)
        mel_loss_before_postnet = torch.sum(mel_loss_before_postnet) / mask_mel_loss.sum().item()

        mel_loss_after_postnet = mask_mel_loss * nn.MSELoss(reduction='none')(mel_out_postnet, mel_target)
        mel_loss_after_postnet = torch.sum(mel_loss_after_postnet) / mask_mel_loss.sum().item()

        print("before", mel_loss_before_postnet.item())
        print("after", mel_loss_after_postnet.item())
        print("gate", gate_loss.item())
        print("attn", attn_loss.item())
        return 5 * (mel_loss_before_postnet + mel_loss_after_postnet) + gate_loss + attn_loss
