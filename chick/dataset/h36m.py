import copy
from argparse import Namespace
from typing import Literal

import numpy as np
import torch.utils.data as data

from chick.dataset.camera import normalize_screen_coordinates, world_to_camera
from chick.dataset.generator import ChunkedGenerator
from chick.dataset.mocap_dataset import MocapDataset
from chick.dataset.skeleton import Skeleton
from chick.utils.reproducibility import deterministic_random

DatasetMode = Literal["train", "eval"]

h36m_skeleton = Skeleton(
    parents=[
        -1,
        0,
        1,
        2,
        3,
        4,
        0,
        6,
        7,
        8,
        9,
        0,
        11,
        12,
        13,
        14,
        12,
        16,
        17,
        18,
        19,
        20,
        19,
        22,
        12,
        24,
        25,
        26,
        27,
        28,
        27,
        30,
    ],
    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
)

cameras = ["54138969", "55011271", "58860488", "60457274"]
h36m_cameras_intrinsic_params = [
    {
        "id": "54138969",
        "center": [512.54150390625, 515.4514770507812],
        "focal_length": [1145.0494384765625, 1143.7811279296875],
        "radial_distortion": [
            -0.20709891617298126,
            0.24777518212795258,
            -0.0030751503072679043,
        ],
        "tangential_distortion": [-0.0009756988729350269, -0.00142447161488235],
        "res_w": 1000,
        "res_h": 1002,
        "azimuth": 70,
    },
    {
        "id": "55011271",
        "center": [508.8486328125, 508.0649108886719],
        "focal_length": [1149.6756591796875, 1147.5916748046875],
        "radial_distortion": [
            -0.1942136287689209,
            0.2404085397720337,
            0.006819975562393665,
        ],
        "tangential_distortion": [-0.0016190266469493508, -0.0027408944442868233],
        "res_w": 1000,
        "res_h": 1000,
        "azimuth": -70,
    },
    {
        "id": "58860488",
        "center": [519.8158569335938, 501.40264892578125],
        "focal_length": [1149.1407470703125, 1148.7989501953125],
        "radial_distortion": [
            -0.2083381861448288,
            0.25548800826072693,
            -0.0024604974314570427,
        ],
        "tangential_distortion": [0.0014843869721516967, -0.0007599993259645998],
        "res_w": 1000,
        "res_h": 1000,
        "azimuth": 110,
    },
    {
        "id": "60457274",
        "center": [514.9682006835938, 501.88201904296875],
        "focal_length": [1145.5113525390625, 1144.77392578125],
        "radial_distortion": [
            -0.198384091258049,
            0.21832367777824402,
            -0.008947807364165783,
        ],
        "tangential_distortion": [-0.0005872055771760643, -0.0018133620033040643],
        "res_w": 1000,
        "res_h": 1002,
        "azimuth": -110,
    },
]

h36m_cameras_extrinsic_params = {
    "S1": [
        {
            "orientation": [
                0.1407056450843811,
                -0.1500701755285263,
                -0.755240797996521,
                0.6223280429840088,
            ],
            "translation": [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        },
        {
            "orientation": [
                0.6157187819480896,
                -0.764836311340332,
                -0.14833825826644897,
                0.11794740706682205,
            ],
            "translation": [1761.278564453125, -5078.0068359375, 1606.2650146484375],
        },
        {
            "orientation": [
                0.14651472866535187,
                -0.14647851884365082,
                0.7653023600578308,
                -0.6094175577163696,
            ],
            "translation": [-1846.7777099609375, 5215.04638671875, 1491.972412109375],
        },
        {
            "orientation": [
                0.5834008455276489,
                -0.7853162288665771,
                0.14548823237419128,
                -0.14749594032764435,
            ],
            "translation": [
                -1794.7896728515625,
                -3722.698974609375,
                1574.8927001953125,
            ],
        },
    ],
    "S2": [
        {},
        {},
        {},
        {},
    ],
    "S3": [
        {},
        {},
        {},
        {},
    ],
    "S4": [
        {},
        {},
        {},
        {},
    ],
    "S5": [
        {
            "orientation": [
                0.1467377245426178,
                -0.162370964884758,
                -0.7551892995834351,
                0.6178938746452332,
            ],
            "translation": [2097.3916015625, 4880.94482421875, 1605.732421875],
        },
        {
            "orientation": [
                0.6159758567810059,
                -0.7626792192459106,
                -0.15728192031383514,
                0.1189815029501915,
            ],
            "translation": [2031.7008056640625, -5167.93310546875, 1612.923095703125],
        },
        {
            "orientation": [
                0.14291371405124664,
                -0.12907841801643372,
                0.7678384780883789,
                -0.6110143065452576,
            ],
            "translation": [-1620.5948486328125, 5171.65869140625, 1496.43701171875],
        },
        {
            "orientation": [
                0.5920479893684387,
                -0.7814217805862427,
                0.1274748593568802,
                -0.15036417543888092,
            ],
            "translation": [-1637.1737060546875, -3867.3173828125, 1547.033203125],
        },
    ],
    "S6": [
        {
            "orientation": [
                0.1337897777557373,
                -0.15692396461963654,
                -0.7571090459823608,
                0.6198879480361938,
            ],
            "translation": [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
        },
        {
            "orientation": [
                0.6147197484970093,
                -0.7628812789916992,
                -0.16174767911434174,
                0.11819244921207428,
            ],
            "translation": [1969.803955078125, -5128.73876953125, 1632.77880859375],
        },
        {
            "orientation": [
                0.1529948115348816,
                -0.13529130816459656,
                0.7646096348762512,
                -0.6112781167030334,
            ],
            "translation": [-1769.596435546875, 5185.361328125, 1476.993408203125],
        },
        {
            "orientation": [
                0.5916101336479187,
                -0.7804774045944214,
                0.12832270562648773,
                -0.1561593860387802,
            ],
            "translation": [-1721.668701171875, -3884.13134765625, 1540.4879150390625],
        },
    ],
    "S7": [
        {
            "orientation": [
                0.1435241848230362,
                -0.1631336808204651,
                -0.7548328638076782,
                0.6188824772834778,
            ],
            "translation": [1974.512939453125, 4926.3544921875, 1597.8326416015625],
        },
        {
            "orientation": [
                0.6141672730445862,
                -0.7638262510299683,
                -0.1596645563840866,
                0.1177929937839508,
            ],
            "translation": [1937.0584716796875, -5119.7900390625, 1631.5665283203125],
        },
        {
            "orientation": [
                0.14550060033798218,
                -0.12874816358089447,
                0.7660516500473022,
                -0.6127139329910278,
            ],
            "translation": [-1741.8111572265625, 5208.24951171875, 1464.8245849609375],
        },
        {
            "orientation": [
                0.5912848114967346,
                -0.7821764349937439,
                0.12445473670959473,
                -0.15196487307548523,
            ],
            "translation": [-1734.7105712890625, -3832.42138671875, 1548.5830078125],
        },
    ],
    "S8": [
        {
            "orientation": [
                0.14110587537288666,
                -0.15589867532253265,
                -0.7561917304992676,
                0.619644045829773,
            ],
            "translation": [2150.65185546875, 4896.1611328125, 1611.9046630859375],
        },
        {
            "orientation": [
                0.6169601678848267,
                -0.7647668123245239,
                -0.14846350252628326,
                0.11158157885074615,
            ],
            "translation": [2219.965576171875, -5148.453125, 1613.0440673828125],
        },
        {
            "orientation": [
                0.1471444070339203,
                -0.13377119600772858,
                0.7670128345489502,
                -0.6100369691848755,
            ],
            "translation": [-1571.2215576171875, 5137.0185546875, 1498.1761474609375],
        },
        {
            "orientation": [
                0.5927824378013611,
                -0.7825870513916016,
                0.12147816270589828,
                -0.14631995558738708,
            ],
            "translation": [-1476.913330078125, -3896.7412109375, 1547.97216796875],
        },
    ],
    "S9": [
        {
            "orientation": [
                0.15540587902069092,
                -0.15548215806484222,
                -0.7532095313072205,
                0.6199594736099243,
            ],
            "translation": [2044.45849609375, 4935.1171875, 1481.2275390625],
        },
        {
            "orientation": [
                0.618784487247467,
                -0.7634735107421875,
                -0.14132238924503326,
                0.11933968216180801,
            ],
            "translation": [1990.959716796875, -5123.810546875, 1568.8048095703125],
        },
        {
            "orientation": [
                0.13357827067375183,
                -0.1367100477218628,
                0.7689454555511475,
                -0.6100738644599915,
            ],
            "translation": [-1670.9921875, 5211.98583984375, 1528.387939453125],
        },
        {
            "orientation": [
                0.5879399180412292,
                -0.7823407053947449,
                0.1427614390850067,
                -0.14794869720935822,
            ],
            "translation": [-1696.04345703125, -3827.099853515625, 1591.4127197265625],
        },
    ],
    "S11": [
        {
            "orientation": [
                0.15232472121715546,
                -0.15442320704460144,
                -0.7547563314437866,
                0.6191070079803467,
            ],
            "translation": [2098.440185546875, 4926.5546875, 1500.278564453125],
        },
        {
            "orientation": [
                0.6189449429512024,
                -0.7600917220115662,
                -0.15300633013248444,
                0.1255258321762085,
            ],
            "translation": [2083.182373046875, -4912.1728515625, 1561.07861328125],
        },
        {
            "orientation": [
                0.14943228662014008,
                -0.15650227665901184,
                0.7681233882904053,
                -0.6026304364204407,
            ],
            "translation": [-1609.8153076171875, 5177.3359375, 1537.896728515625],
        },
        {
            "orientation": [
                0.5894251465797424,
                -0.7818877100944519,
                0.13991211354732513,
                -0.14715361595153809,
            ],
            "translation": [-1590.738037109375, -3854.1689453125, 1578.017578125],
        },
    ],
}


class Human36mDataset(MocapDataset):
    def __init__(self, path, crop_uv=0, remove_static_joints=True):
        super().__init__(fps=50, skeleton=h36m_skeleton)
        self.train_list = ["S1", "S5", "S6", "S7", "S8"]
        self.test_list = ["S9", "S11"]

        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        for cameras in self._cameras.values():
            for i, cam in enumerate(cameras):
                cam.update(h36m_cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ["id", "res_w", "res_h"]:
                        cam[k] = np.array(v, dtype="float32")

                if crop_uv == 0:
                    cam["center"] = normalize_screen_coordinates(
                        cam["center"], w=cam["res_w"], h=cam["res_h"]
                    ).astype("float32")
                    cam["focal_length"] = cam["focal_length"] / cam["res_w"] * 2

                if "translation" in cam:
                    cam["translation"] = cam["translation"] / 1000

                cam["intrinsic"] = np.concatenate(
                    (
                        cam["center"],
                        cam["focal_length"],
                        cam["radial_distortion"],
                        cam["tangential_distortion"],
                    )
                )

        data = np.load(path, allow_pickle=True)["positions_3d"].item()

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    "positions": positions,
                    "cameras": self._cameras[subject],
                }

        if remove_static_joints:
            self.remove_joints(
                [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]
            )

            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8

    def supports_semi_supervised(self):
        return True


class Fusion(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True, skip=10):
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(",")
        self.test_list = opt.subjects_test.split(",")
        self.action_filter = None if opt.actions == "*" else opt.actions.split(",")
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
        self.skip = skip

        if self.train:
            self.keypoints = self.prepare_data(dataset, self.train_list)
            self.cameras_train, self.poses_train, self.poses_train_2d = self.fetch(
                dataset, self.train_list, subset=self.subset
            )

            # if opt.batch_size < opt.stride:
            #     raise ValueError('Batch size must be greater or equal to the stride')

            self.generator = ChunkedGenerator(
                opt.batch_size // opt.stride,
                self.cameras_train,
                self.poses_train,
                self.poses_train_2d,
                self.stride,
                pad=self.pad,
                kps_left=self.kps_left,
                kps_right=self.kps_right,
                joints_left=self.joints_left,
                joints_right=self.joints_right,
                out_all=opt.out_all,
                augment=True,
            )
            print("INFO: Training on {} frames".format(self.generator.num_frames()))
        else:
            self.keypoints = self.prepare_data(dataset, self.test_list)
            self.cameras_test, self.poses_test, self.poses_test_2d = self.fetch(
                dataset, self.test_list, subset=self.subset
            )
            self.generator = ChunkedGenerator(
                opt.batch_size // opt.stride,
                self.cameras_test,
                self.poses_test,
                self.poses_test_2d,
                self.stride,
                pad=self.pad,
                kps_left=self.kps_left,
                kps_right=self.kps_right,
                joints_left=self.joints_left,
                joints_right=self.joints_right,
                out_all=opt.out_all,
            )
            self.key_index = self.generator.saved_index

    def prepare_data(self, dataset, folder_list):
        for subject in folder_list:
            for action in dataset[subject].keys():
                anim = dataset[subject][action]

                positions_3d = []
                for cam in anim["cameras"]:
                    pos_3d = world_to_camera(
                        anim["positions"], R=cam["orientation"], t=cam["translation"]
                    )
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim["positions_3d"] = positions_3d

        keypoints = np.load(
            self.root_path
            + "data_2d_"
            + self.data_type
            + "_"
            + self.keypoints_name
            + ".npz",
            allow_pickle=True,
        )
        keypoints_symmetry = keypoints["metadata"].item()["keypoints_symmetry"]

        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(
            keypoints_symmetry[1]
        )
        self.joints_left, self.joints_right = list(
            dataset.skeleton().joints_left()
        ), list(dataset.skeleton().joints_right())
        keypoints = keypoints["positions_2d"].item()

        for subject in folder_list:
            assert (
                subject in keypoints
            ), "Subject {} is missing from the 2D detections dataset".format(subject)
            for action in dataset[subject].keys():
                assert (
                    action in keypoints[subject]
                ), "Action {} of subject {} is missing from the 2D detections dataset".format(
                    action, subject
                )
                for cam_idx in range(len(keypoints[subject][action])):

                    mocap_length = dataset[subject][action]["positions_3d"][
                        cam_idx
                    ].shape[0]
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        keypoints[subject][action][cam_idx] = keypoints[subject][
                            action
                        ][cam_idx][:mocap_length]

        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    cam = dataset.cameras()[subject][cam_idx]
                    if self.crop_uv == 0:
                        kps[..., :2] = normalize_screen_coordinates(
                            kps[..., :2], w=cam["res_w"], h=cam["res_h"]
                        )
                    keypoints[subject][action][cam_idx] = kps

        return keypoints

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = self.keypoints[subject][action]

                for i in range(len(poses_2d)):
                    out_poses_2d[(subject, action, i)] = poses_2d[i]

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), "Camera count mismatch"
                    for i, cam in enumerate(cams):
                        if "intrinsic" in cam:
                            out_camera_params[(subject, action, i)] = cam["intrinsic"]

                if parse_3d_poses and "positions_3d" in dataset[subject][action]:
                    poses_3d = dataset[subject][action]["positions_3d"]
                    assert len(poses_3d) == len(poses_2d), "Camera count mismatch"
                    for i in range(len(poses_3d)):
                        out_poses_3d[(subject, action, i)] = poses_3d[i]

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.downsample
        if subset < 1:
            for key in out_poses_2d.keys():
                n_frames = int(
                    round(len(out_poses_2d[key]) // stride * subset) * stride
                )
                start = deterministic_random(
                    0,
                    len(out_poses_2d[key]) - n_frames + 1,
                    str(len(out_poses_2d[key])),
                )
                out_poses_2d[key] = out_poses_2d[key][start : start + n_frames : stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][
                        start : start + n_frames : stride
                    ]
        elif stride > 1:
            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d

    def __len__(self):
        return len(self.generator.pairs) // self.skip

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[
            index * self.skip
        ]

        cam, gt_3D, input_2D, action, subject, cam_ind = self.generator.get_batch(
            seq_name, start_3d, end_3d, flip, reverse
        )

        if self.train == False and self.test_aug:
            _, _, input_2D_aug, _, _, _ = self.generator.get_batch(
                seq_name, start_3d, end_3d, flip=True, reverse=reverse
            )
            input_2D = np.concatenate(
                (
                    np.expand_dims(input_2D, axis=0),
                    np.expand_dims(input_2D_aug, axis=0),
                ),
                0,
            )

        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = np.float64(1.0)

        return (
            cam,
            gt_3D,
            input_2D_update,
            action,
            subject,
            scale,
            bb_box,
            cam_ind,
            start_3d,
        )


class H36MVideoDataset(Fusion):
    def __init__(
        self,
        path,
        root_path,
        frames=1,
        mode: DatasetMode = "eval",
        config=None,
        keypoints="gt",
    ):
        self.config = {
            "actions": "*",
            "batch_size": 64,
            "crop_uv": 0,
            "dataset": "h36m",
            "downsample": 1,
            "frames": frames,
            "keypoints": keypoints,
            "out_all": 1,
            "skip": 1,
            "stride": 1,
            "subjects_train": "S1,S5,S6,S7,S8",
            "subjects_test": "S9,S11",
            "subset": 1,
            "test_augmentation": False,
            "reverse_augmentation": False,
        }

        self.train = True if mode == "train" else False

        self.config["pad"] = (self.config["frames"] - 1) / 2

        if config is not None:
            self.config.update(config)

        self.config = Namespace(**self.config)

        self.dataset = Human36mDataset(path, crop_uv=self.config.crop_uv)
        super(H36MVideoDataset, self).__init__(
            opt=self.config, train=self.train, dataset=self.dataset, root_path=root_path
        )
