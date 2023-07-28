import numpy as np
import torch
from collections import OrderedDict

from propose.propose.models.detectors.hrnet import HRNet
from propose.propose.models.detectors.hrnet.config import config
from tqdm import tqdm

config_file = "./models/w32_256x256_adam_lr1e-3.yaml"

config.defrost()
config.merge_from_file(config_file)
config.freeze()

hrnet = HRNet(config)

state_dict = torch.load(
    "./models" + "/fine_HRNet.pt",
    map_location=torch.device("cpu"),
)["net"]

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k  # remove module.
    new_state_dict[name] = v

hrnet.load_state_dict(new_state_dict, strict=False)


def get_heatmaps(image: torch.Tensor, model: HRNet):
    """
    It takes an image and a model, and returns the heatmaps

    :param image: a tensor of shape (1, 3, 256, 256)
    :type image: torch.Tensor
    :param model: The HRNet model
    :type model: HRNet
    :return: A list of heatmaps.
    """
    model.eval()

    with torch.no_grad():
        heatmaps = model(image)
        return heatmaps


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    from torchvision import transforms

    # image = Image.open("./Walking.58860488.764.png")
    image = Image.open("./img.png")
    image = transforms.ToTensor()(image)
    image = transforms.Resize((256, 256))(image)
    image = image.unsqueeze(0)

    heatmaps = get_heatmaps(image, model)
    print(heatmaps.shape)
    heatmaps = heatmaps.detach()
    heatmaps = heatmaps.squeeze(0)

    coords, conf = model.get_max_preds(heatmaps.unsqueeze(0).numpy())
    print(coords)
    features = np.concatenate([coords, conf], axis=-1).squeeze()
    features = torch.from_numpy(features)
    #
    # # fig, axs = plt.subplots(4, 4)
    # # for i in range(4):
    # #     for j in range(4):
    # #         axs[i, j].imshow(heatmaps[i * 4 + j])
    # # plt.show()
    #
    # # Embed the heatmaps using Vision Transformer from pytorch
    # import timm
    #
    # model = timm.create_model('vit_base_patch16_224', num_classes=3)
    # model.eval()
    #
    # # Resize the heatmaps to 224x224
    # heatmaps = transforms.Resize((224, 224))(heatmaps)
    #
    # # Add a batch dimension
    # heatmaps = heatmaps.unsqueeze(0).permute(1, 0, 2, 3).repeat(1, 3, 1, 1)
    #
    # optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    #
    # pbar = tqdm(range(1000))
    # for i in pbar:
    #     optim.zero_grad()
    #     output = model(heatmaps)
    #     loss = torch.nn.functional.mse_loss(output, features)
    #     loss.backward()
    #     optim.step()
    #     pbar.set_description(f"Loss: {loss.item()} | {output[0]} | {features[0]}")
    #
    # print(output)
    # print(features)







