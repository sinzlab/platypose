import os
from collections import OrderedDict

import numpy as np
import torch
import wandb

from .config import config
from .models.pose_hrnet import PoseHighResolutionNet
from .utils import crop_image_to_human
from .fit_gaussians import fit_single_gaussian


class HRNet(PoseHighResolutionNet):
    @classmethod
    def from_pretrained(cls, artifact_name=None, config_file=None, **kwargs) -> "HRNet":
        if not config_file:
            dirname = os.path.dirname(__file__)
            config_file = os.path.join(
                dirname, "experiments/w32_256x256_adam_lr1e-3.yaml"
            )

            config.defrost()
            config.merge_from_file(config_file)
            config.freeze()

        model = cls(config, **kwargs)

        api = wandb.Api()
        artifact = api.artifact(artifact_name, type="model")

        if wandb.run:
            wandb.run.use_artifact(artifact, type="model")

        artifact_dir = artifact.download()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(
            artifact_dir + "/pose_hrnet_w32_256x256.pth",
            map_location=torch.device(device),
        )

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k  # remove module.
            #  print(name,'\t')
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)

        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def get_max_preds(batch_heatmaps: np.array) -> tuple[np.array, np.array]:
        """
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        """
        assert isinstance(
            batch_heatmaps, np.ndarray
        ), "batch_heatmaps should be numpy.ndarray"
        assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals

    def pose_estimate(self, input: torch.Tensor) -> np.array:
        batch_heatmaps = self.forward(input)

        coords, maxvals = self.get_max_preds(batch_heatmaps.cpu().detach().numpy())

        params = []
        for i in range(16):
            param = fit_single_gaussian(batch_heatmaps[0, i].cpu(), coords[0, i])
            params.append(param)

        params = np.stack(params)
        params[:, :2] *= 4
        params[:, 2:] *= 16

        return params

    @classmethod
    def preprocess(
        cls, images: torch.Tensor, detector: torch.nn.Module = None
    ) -> torch.Tensor:
        if detector is None:
            detector = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

        detector.eval()

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        cropped_images = []
        bboxes = []
        scales = []
        for image in images:
            cropped_image, bbox, scale = crop_image_to_human(image, detector)
            cropped_images.append(torch.Tensor(cropped_image))
            bboxes.append(bbox)
            scales.append(scale)

        cropped_images = torch.stack(cropped_images)
        pred_image = cropped_images.permute(0, 3, 1, 2)

        return pred_image, bboxes, scales

    @classmethod
    def postprocess(cls, coords, bbox, scale):
        return coords / scale + np.array([bbox[0], bbox[2]])
