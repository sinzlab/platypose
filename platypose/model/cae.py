import torch
import torch.nn as nn


class CAE(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        batch["x_xyz"] = batch["x"]
        # encode
        batch.update(self.encoder(batch))
        # decode
        batch.update(self.decoder(batch))
        # if we want to output xyz
        batch["output_xyz"] = batch["output"]
        return batch
