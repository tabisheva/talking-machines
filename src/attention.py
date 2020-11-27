import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationBlock(nn.Module):

    def __init__(
            self,
            attention_n_filters,
            attention_kernel_size,
            attention_dim
    ):
        super().__init__()

        padding = int((attention_kernel_size - 1) / 2)
        self.conv = nn.Conv1d(
            2, attention_n_filters, kernel_size=attention_kernel_size,
            padding=padding, bias=False, stride=1, dilation=1
        )
        self.projection = nn.Linear(attention_n_filters, attention_dim, bias=False)

    def forward(self, attention_weights):
        output = self.conv(attention_weights).transpose(1, 2)
        output = self.projection(output)
        return output


class Attention(nn.Module):

    def __init__(
            self,
            attention_rnn_dim,
            embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size
    ):
        super().__init__()

        self.query = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationBlock(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(
            self,
            query,
            processed_memory,
            attention_weights_cat
    ):
        """
        query: decoder output (batch, n_mel_channels)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        """
        processed_query = self.query(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)

        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory
        ))

        energies = energies.squeeze(2)
        return energies

    def forward(
            self,
            attention_hidden_state,
            memory,
            processed_memory,
            attention_weights_cat,
            mask,
    ):
        """
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """

        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights

class MonotonicAttention(Attention):
    # https://github.com/j-min/MoChA-pytorch
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def gaussian_noise(self, tensor_like):
        return torch.empty_like(tensor_like).normal_()

    def safe_cumprod(self, x):
        return torch.clamp(torch.exp(torch.cumsum(torch.log(torch.clamp(x, min=1e-10, max=1)), dim=1)), min=1e-45)

    def exclusive_cumprod(self, x):
        batch_size, sequence_length = x.size()
        if torch.cuda.is_available():
            one_x = torch.cat([torch.ones(batch_size, 1).cuda(), x], dim=1)[:, :-1]
        else:
            one_x = torch.cat([torch.ones(batch_size, 1), x], dim=1)[:, :-1]
        return torch.cumprod(one_x, dim=1)

    def forward(self,
        attention_hidden_state,
        memory,
        processed_memory,
        attention_weights_cat,
        mask
    ):

        if attention_weights_cat.sum() == 0:
            alpha = torch.zeros_like(attention_weights_cat[:, 0], requires_grad=True)
            alpha[:, 0] = 1.
            attention_context = torch.bmm(alpha.unsqueeze(1), memory)
            attention_context = attention_context.squeeze(1)
            return attention_context, alpha
        else:
            alignment = super().get_alignment_energies(
                    attention_hidden_state, processed_memory, attention_weights_cat
            )
            if self.training:
                if mask is not None:
                    alignment = alignment.data.masked_fill_(mask, self.score_mask_value)
                p_select = self.sigmoid(alignment + self.gaussian_noise(alignment))
                cumprod_1_minus_p = self.safe_cumprod(1 - p_select)
                alpha = p_select * cumprod_1_minus_p * \
                    torch.cumsum(attention_weights_cat[:, 0] / cumprod_1_minus_p, dim=1)
                attention_context = torch.bmm(alpha.unsqueeze(1), memory)
                attention_context = attention_context.squeeze(1)
                return attention_context, alpha
            else:
                above_threshold = (alignment > 0).float()
                p_select = above_threshold * torch.cumsum(attention_weights_cat[:, 0], dim=1)
                attention = p_select * self.exclusive_cumprod(1 - p_select)

                attended = attention.sum(dim=1)
                for batch_i in range(attention_weights_cat.shape[0]):
                    if not attended[batch_i]:
                        attention[batch_i, -1] = 1
                attention_context = torch.bmm(attention.unsqueeze(1), memory)
                attention_context = attention_context.squeeze(1)
                return attention_context, attention