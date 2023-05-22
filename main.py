import random

from chick.train.train_platforms import NoPlatform
from chick.train.training_loop import TrainLoop
from chick.utils.fixseed import fixseed
from chick.utils.model_util import create_model_and_diffusion
from chick.utils.parser_util import train_args
from common.h36m_dataset import Human36mDataset
from common.load_data_hm36 import Fusion
from common.opt import opts
from common.utils import *

opt = opts().parse()
args = train_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


if __name__ == "__main__":
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # Check 2D keypoint type
    print(f"Using {opt.keypoints} 2D keypoints")

    root_path = opt.root_path
    dataset_path = root_path + "data_3d_" + opt.dataset + ".npz"

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    train_data = Fusion(
        opt=opt, train=True, dataset=dataset, root_path=root_path, skip=1
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name="Args")

    model, diffusion = create_model_and_diffusion(args)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model.to(device)

    print(
        "Number of parameters: {:,}".format(
            sum(p.numel() for p in model.parameters_wo_clip() if p.requires_grad)
        )
    )

    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, train_dataloader).run_loop()

    # Loss: 0.03779 U-Net like model
