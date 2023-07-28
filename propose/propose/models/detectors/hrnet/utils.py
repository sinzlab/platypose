import torch
from skimage.transform import rescale
from torchvision.transforms import Pad


def crop_image_to_human(input_image, detector):
    if isinstance(input_image, torch.Tensor):
        input_image = input_image.numpy()

    detections = detector(input_image)
    detections = (
        detections.pandas()
        .xyxy[0][detections.pandas().xyxy[0].name == "person"]
        .reset_index()
    )
    bbox = detections.iloc[0]

    xy = (bbox["xmin"], bbox["ymax"])
    width = bbox["xmax"] - bbox["xmin"]
    height = bbox["ymax"] - bbox["ymin"]

    center = (xy[0] + width / 2, xy[1] - height / 2)

    side = max([width, height]) + 10

    crop_size = [
        int(center[0] - side / 2),
        int(center[0] + side / 2),
        int(center[1] - side / 2),
        int(center[1] + side / 2),
    ]
    for i in range(4):
        crop_size[i] = max([crop_size[i], 0])

    cropped_image = input_image[
        crop_size[2] : crop_size[3], crop_size[0] : crop_size[1]
    ]

    padder = Pad(
        (
            int((max(cropped_image.shape) - cropped_image.shape[0]) / 2),
            int((max(cropped_image.shape) - cropped_image.shape[1]) / 2),
        )
    )
    cropped_image = padder(torch.Tensor(cropped_image)).numpy()
    cropped_image = cropped_image / 255

    cropped_image = rescale(
        cropped_image, 256 / cropped_image.shape[0], channel_axis=-1
    )
    cropped_image = cropped_image[:256, :256]
    padder = Pad(
        (
            256 - cropped_image.shape[0],
            256 - cropped_image.shape[1],
        )
    )

    cropped_image = padder(torch.Tensor(cropped_image)).numpy()
    cropped_image = cropped_image[:256, :256]

    return cropped_image
