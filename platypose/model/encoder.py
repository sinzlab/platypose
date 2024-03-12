import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class SkeletonEncoder(nn.Module):
    def __init__(
        self,
        njoints=17,
        nfeats=3,
        latent_dim=256,
        ff_size=1024,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        **kwargs
    ):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation

        self.input_feats = self.njoints * self.nfeats

        self.muQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The function takes in a tensor of shape (batch_size, num_joints, num_features, num_frames) and returns a tensor of
        shape (num_frames, batch_size, num_joints * num_features)

        :param x: the input skeleton sequence, of shape (batch_size, num_joints, num_features, num_frames)
        :type x: torch.Tensor
        :return: The mean of the distribution
        """
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        # embedding of the skeleton
        x = self.skelEmbedding(x)

        muQuery = self.muQuery.unsqueeze(0).repeat(1, bs, 1)
        sigmaQuery = self.sigmaQuery.unsqueeze(0).repeat(1, bs, 1)

        xseq = torch.cat((muQuery, sigmaQuery, x), axis=0)

        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        mu = self.seqTransEncoder(xseq)

        return mu
