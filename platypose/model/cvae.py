import torch
from common.h36m_dataset import Human36mDataset
from common.load_data_hm36 import Fusion
from common.opt import opts

from platypose.model.cae import CAE


class CVAE(CAE):
    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def forward(self, batch):
        batch["x_xyz"] = batch["x"]
        # encode
        batch.update(self.encoder(batch))
        batch["z"] = self.reparameterize(batch)

        # decode
        batch.update(self.decoder(batch))

        # if we want to output xyz
        batch["output_xyz"] = batch["output"]

        return batch

    def return_latent(self, batch, seed=None):
        distrib_param = self.encoder(batch)
        batch.update(distrib_param)
        return self.reparameterize(batch, seed=seed)

    @classmethod
    def build(cls):
        from platypose.model.transformer import Decoder_TRANSFORMER, Encoder_TRANSFORMER

        encoder = Encoder_TRANSFORMER()
        decoder = Decoder_TRANSFORMER()

        return cls(encoder, decoder)

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path):
        """
        `from_pretrained` is a class method that takes a path to a pretrained model and returns a model with the same
        architecture and weights as the pretrained model

        :param cls: the class of the model
        :param path: path to the pretrained model
        :return: The model is being returned.
        """
        model = cls.build()
        model.load_state_dict(torch.load(path))
        return model

    def kl_loss(self, batch):
        """
        The KL divergence loss is the sum of the difference between the log of the variance and the log of the standard
        deviation, plus the difference between the mean and the variance, all divided by two

        :param batch: a dictionary of tensors, each of which is a batch of data
        :return: The KL divergence between the prior and the posterior.
        """
        mu, logvar = batch["mu"], batch["logvar"]
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def recon_loss(self, batch):
        """
        It takes in a batch of data, and returns the mean squared error between the output of the model and the input to the
        model

        :param batch: a dictionary containing the following keys:
        :return: The reconstruction loss is being returned.
        """
        return torch.nn.functional.mse_loss(batch["output"], batch["x"])

    def _loss(self, batch, kl_weight=1e-5):
        """
        The loss function is the sum of the mean squared error loss and the KL divergence loss

        :param batch: a batch of data, which is a dictionary of the form {'inputs': [batch_size, input_size], 'targets':
        [batch_size, output_size]}
        :param kl_weight: The weight of the KL loss
        :return: The loss function is being returned.
        """
        return self.recon_loss(batch) + kl_weight * self.kl_loss(batch)

    def elbo(self, batch, kl_weight=1e-5):
        """
        The function takes in a batch of data, and returns the loss of the model on that batch

        :param batch: a batch of data
        :param kl_weight: The weight of the KL divergence term in the ELBO
        :return: The loss of the batch
        """
        batch = self.forward(batch)
        return self._loss(batch, kl_weight=kl_weight)

    def step(self, batch, opt):
        opt.zero_grad()
        loss = self.elbo(batch)
        loss.backward()
        opt.step()

        return loss

    def train(self, dataset):
        from tqdm import tqdm

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4)

        pbar = tqdm(enumerate(dataloader))
        for i, b in pbar:
            gt_3D = b[1]
            gt_3D = gt_3D.permute(0, 2, 3, 1)

            batch = {
                "x": gt_3D,
                "mask": torch.ones(gt_3D.shape[0], gt_3D.shape[-1], dtype=torch.bool),
                "lengths": torch.ones(gt_3D.shape[0], dtype=torch.long)
                * gt_3D.shape[-1],
            }

            loss = self.step(batch, opt)
            pbar.set_description(f"loss: {loss.item():.4f}")

            if i % 1000 == 0:
                self.save("./cvae_checkpoint.pth")


opt = opts().parse()

if __name__ == "__main__":
    opt.manualSeed = 1
    torch.manual_seed(opt.manualSeed)

    # Check 2D keypoint type
    print(f"Using {opt.keypoints} 2D keypoints")

    root_path = opt.root_path
    dataset_path = "../../" + root_path + "data_3d_" + opt.dataset + ".npz"

    dataset = Human36mDataset(dataset_path, opt)
    dataset = Fusion(
        opt=opt, train=True, dataset=dataset, root_path="../../" + root_path
    )

    model = CVAE.build()
    model.train(dataset)


# if __name__ == '__main__':
#     cvae = CVAE.build()
#
#     frames = 30
#     batch = {
#         "x": torch.randn(10, 17, 3, frames),
#         "mask": torch.randn(10, frames).bool(),
#         "lengths": torch.ones(10) * frames,
#     }
#
#     cvae.elbo(batch)
