import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from model.rotation2xyz import Rotation2xyz
from platypose.model.encoder import SkeletonEncoder
from platypose.model.transformer import MartinezTransformerEncoderLayer, TransformerUNet


class MDM(nn.Module):
    def __init__(
        self,
        modeltype,
        njoints,
        nfeats,
        num_actions,
        translation,
        pose_rep,
        glob,
        glob_rot,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        ablation=None,
        activation="gelu",
        legacy=False,
        data_rep="rot6d",
        dataset="amass",
        clip_dim=512,
        arch="trans_enc",
        emb_trans_dec=False,
        clip_version=None,
        **kargs
    ):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get("action_emb", None)

        self.input_feats = 51  # 166 #self.njoints * self.nfeats

        self.normalize_output = kargs.get("normalize_encoder_output", False)

        self.cond_mode = "no_cond"  # kargs.get('cond_mode', 'no_cond')
        # print('cond_mode', self.cond_mode)
        self.cond_mask_prob = kargs.get("cond_mask_prob", 0.0)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == "gru" else 0
        self.input_process = InputProcess(
            self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim
        )

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec

        self.skeleton_encoder = SkeletonEncoder(latent_dim=self.latent_dim)

        if self.arch == "trans_enc":
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
        elif self.arch == "trans_unet":
            print("TRANS_UNET init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation,
            )

            self.seqTransEncoder = TransformerUNet(
                seqTransEncoderLayer, num_layers=self.num_layers
            )

        elif self.arch == "trans_dec":
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(
                d_model=self.latent_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=activation,
            )
            self.seqTransDecoder = nn.TransformerDecoder(
                seqTransDecoderLayer, num_layers=self.num_layers
            )
        elif self.arch == "gru":
            print("GRU init")
            self.gru = nn.GRU(
                self.latent_dim,
                self.latent_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
        else:
            raise ValueError(
                "Please choose correct architecture [trans_enc, trans_dec, gru]"
            )

        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        if self.cond_mode != "no_cond":
            if "text" in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print("EMBED TEXT")
            if "action" in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print("EMBED ACTION")

        self.output_process = OutputProcess(
            self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats
        )

        # self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [
            p
            for name, p in self.named_parameters()
            if not name.startswith("clip_model.")
        ]

    def mask_cond(self, cond, force_mask=False):
        frames, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(
                bs, 1
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, nframes, nfeats = x.shape
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get("uncond", False)
        if "text" in self.cond_mode:
            enc_text = self.encode_text(y["text"])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if "action" in self.cond_mode:
            action_emb = self.embed_action(y["action"])
            emb += self.mask_cond(action_emb, force_mask=force_mask)
        if "skeleton" in self.cond_mode:
            enc_skeleton = self.skeleton_encoder(y["skeleton"])
            emb = self.mask_cond(enc_skeleton, force_mask=force_mask) + emb

        # if self.arch == "gru":
        #     x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)
        #     emb_gru = emb.repeat(nframes, 1, 1)  # [#frames, bs, d]
        #     emb_gru = emb_gru.permute(1, 2, 0)  # [bs, d, #frames]
        #     emb_gru = emb_gru.reshape(
        #         bs, self.latent_dim, 1, nframes
        #     )  # [bs, d, 1, #frames]
        #     x = torch.cat(
        #         (x_reshaped, emb_gru), axis=1
        #     )  # [bs, d+joints*feat, 1, #frames]

        x = self.input_process(x)

        if self.arch == "trans_enc":
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]

            # if in train mode
            # if self.training:
            #     mask_prob = 0.9 # rate at which the mask is applied otherwise full sequence is used
            #
            #     if torch.rand(1) < mask_prob and xseq.shape[0] > 8:
            #         # block masking of xseq, set random block lengths to 0
            #         maskseq_length = torch.randint(low=int(xseq.shape[0] * 0.7), high=int(xseq.shape[0] * 0.9), size=(1,)).item()
            #         masksq_start = torch.randint(low=1, high=xseq.shape[0] - maskseq_length, size=(1,)).item()
            #
            #         xseq[masksq_start:masksq_start+maskseq_length] = 0

            output = self.seqTransEncoder(xseq)[
                1:
            ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == "trans_unet":
            # adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[
                1:
            ]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == "trans_dec":
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[
                    1:
                ]  # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == "gru":
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]
            output, _ = self.gru(xseq)

        output = self.output_process(output)

        output = output[:, -nframes:]  # [bs, njoints, nfeats, nframes]

        return output

    def _apply(self, fn):
        super()._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)


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


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim

        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == "rot_vel":
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # (bs, nframes, nfeats)
        # bs, njoints, nfeats, nframes = x.shape
        # x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        x = x.permute((1, 0, 2))

        if self.data_rep in ["rot6d", "xyz", "hml_vec", "h36m"]:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == "rot_vel":
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == "rot_vel":
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ["rot6d", "xyz", "hml_vec", "h36m"]:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == "rot_vel":
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.permute(
            (1, 0, 2)
        )  # (nframes, bs, njoints * nfeats) -> (bs, nframes, nfeats)
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        # output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output