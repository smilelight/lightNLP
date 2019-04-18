import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DEVICE, DEFAULT_CONFIG


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lens, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.size(0)
        max_len = trg.size(1)
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(DEVICE)

        encoder_output, hidden = self.encoder(src, src_lens)
        hidden = hidden[:self.decoder.n_layers]
        decoder_input = trg.data[:, 0]  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                decoder_input, hidden, encoder_output)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            if teacher_force:
                decoder_input = trg.data[:, t].clone().detach().to(DEVICE)
            else:
                decoder_input = top1.to(DEVICE)
        return outputs

    def predict(self, src, src_lens, sos, max_len):
        batch_size = src.size(0)
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(DEVICE)

        encoder_output, hidden = self.encoder(src, src_lens)
        hidden = hidden[:self.decoder.n_layers]
        decoder_input = sos  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                decoder_input, hidden, encoder_output)
            outputs[t] = output
            top1 = output.data.max(1)[1]
            decoder_input = top1.to(DEVICE)
        return outputs
