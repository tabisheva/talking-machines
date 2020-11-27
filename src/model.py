import torch
import torch.nn as nn
import torch.nn.functional as F
from config import PostnetConfig
from src.attention import Attention, MonotonicAttention
from src.preprocessing import get_mask_from_lengths


class Encoder(nn.Module):

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.embedding_size = 512
        self.embedding = nn.Embedding(params.input_dim, 512)
        convolutions = []
        for _ in range(3):
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels=512,
                          out_channels=512,
                          kernel_size=5,
                          padding=2),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        x = self.embedding(x).transpose(1, 2)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5)
        x = x.transpose(1, 2)
        # B x T x 512
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs
    #
    # def inference(self, x):
    #     for conv in self.convolutions:
    #         x = F.dropout(F.relu(conv(x)), 0.5, training=False)
    #     x = x.transpose(1, 2)
    #     outputs, _ = self.lstm(x)
    #     return outputs


class Prenet(nn.Module):
    def __init__(self):
        super(Prenet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(80, 256), nn.Linear(256, 256)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    '''Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    '''

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        postnet_config = PostnetConfig()
        for conv in postnet_config.params:
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=conv["in_channels"],
                              out_channels=conv["out_channels"],
                              kernel_size=5,
                              padding=2),
                    nn.BatchNorm1d(conv["out_channels"])))

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x


class Decoder(nn.Module):
    def __init__(self, params, device):
        super(Decoder, self).__init__()
        self.device = device
        self.prenet = Prenet()
        self.attention_rnn = nn.LSTMCell(256 + 512, 1024)
        self.attention_layer = Attention(1024, 512, 128, 32, 31)
        self.decoder_rnn = nn.LSTMCell(1024 + 512, 1024)
        self.linear_projection = nn.Linear(1024 + 512, 80)
        self.gate_layer = nn.Linear(1024 + 512, 1)

    def decode(self, decoder_input):
        ''' Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output after prenet
        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        '''
        # concatenated prev mel and attention_context vector (B, 256 + 512)
        cell_input = torch.cat((decoder_input, self.attention_context), -1)

        # first LSTMCell with hidden_size 1024
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))

        self.attention_hidden = F.dropout(self.attention_hidden, 0.1)

        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, self.mask)
#        print(self.attention_weights)
        self.attention_weights_cum += self.attention_weights

        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)

        # Second LSTMCell with hidden_size 1024
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, 0.1)

        # linear layer for mel prediction
        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        # binary classifier for stop token
        gate_prediction = torch.sigmoid(self.gate_layer(decoder_hidden_attention_context))
        return decoder_output, gate_prediction, self.attention_weights

    def initialize_decoder_states(self, memory, mask):
        batch_size = memory.size(0)
        num_frames = memory.size(1)
        self.attention_context = torch.zeros((batch_size, 512)).to(self.device)
        self.attention_hidden = torch.zeros((batch_size, 1024)).to(self.device)
        self.attention_cell = torch.zeros((batch_size, 1024)).to(self.device)
        self.decoder_hidden = torch.zeros((batch_size, 1024)).to(self.device)
        self.decoder_cell = torch.zeros((batch_size, 1024)).to(self.device)
        self.attention_weights = torch.zeros((batch_size, num_frames), requires_grad=True).to(self.device)
        self.attention_weights_cum = torch.zeros((batch_size, num_frames)).to(self.device)
        self.memory = memory
        self.processed_memory = self.attention_layer.memory(memory)
        self.mask = mask

    def forward(self, memory, decoder_inputs, memory_lengths):
        """

        :param memory: encoder outputs (B, T, 512)
        :param decoder_inputs: mel from previous step (B, num_mels, T)
        :param memory_lengths: (B, )
        :return:
        """
        # start mel frame with zeros (1, B, num_mels)
        decoder_input = torch.zeros((1, memory.size(0), 80)).to(self.device)
        # (B, num_mels, T) -> (T, B, num_mels)
        decoder_inputs = decoder_inputs.permute(2, 0, 1)
        # (T + 1, B, num_mels)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)
        # (T, B, 256)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths, self.device))

        mel_outputs, gate_outputs, alignments = [], [], []
        for i in range(decoder_inputs.size(0) - 1):
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]
        alignments = torch.stack(alignments).transpose(0, 1)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, 80)
        # (B, T_out, num_mels) -> (B, num_mels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, memory_lengths):
        """

        :param memory: encoder outputs (B, T, 512)
        :param memory_lengths: (B, )
        :return:
        """
        # start mel frame with zeros (1, B, num_mels)
        decoder_input = torch.zeros((1, memory.size(0), 80)).to(self.device).squeeze(0)
        # (B, num_mels, T) -> (T, B, num_mels)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths, self.device))

        mel_outputs, gate_outputs, alignments = [], [], []
        for i in range(1000):
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(0)]
            alignments += [attention_weights]
            if torch.sigmoid(gate_output).item() > 0.6:
                print('Terminated by gate', torch.sigmoid(gate_output).item())
                break
            decoder_input = mel_output
        alignments = torch.stack(alignments).transpose(0, 1)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, 80)
        # (B, T_out, num_mels) -> (B, num_mels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):

    def __init__(self, params, device):
        super(Tacotron2, self).__init__()
        self.device = device
        self.num_mels = 80
        self.encoder = Encoder(params)
        self.decoder = Decoder(params, device)
        self.postnet = Postnet()

    def forward(self, inputs):
        text_inputs, text_lengths, mels, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        encoder_outputs = self.encoder(text_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]

    def inference(self, inputs):
        text_inputs, text_lengths, mels, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        encoder_outputs = self.encoder(text_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = [x.to(self.device) for x in batch]
        return ((text_padded, input_lengths, mel_padded, output_lengths),
                (mel_padded, gate_padded))
