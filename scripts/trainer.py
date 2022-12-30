import hydra
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist

from train.net import Net
from train.gan import GAN


@hydra.main(config_path="../configs/", config_name="train.yaml")
def main(cfg):
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(cfg.train.common.seed)

    if torch.cuda.device_count() > 1:
        print("Train with multiple GPUs")
        cfg.train.ddp.distributed = True
        gpus = torch.cuda.device_count()
        cfg.train.ddp.world_size = gpus * cfg.train.ddp.nodes

        trainer = None
        if cfg.train.common.method == "NET":
            trainer = Net
        elif cfg.train.common.method == "GAN":
            trainer = GAN

        mp.spawn(
            trainer,
            nprocs=gpus,
            args=(cfg,),
        )
        dist.destroy_process_group()
    else:
        print("Train with single GPUs")
        if cfg.train.common.method == "NET":
            Net(0, cfg)
        elif cfg.train.common.method == "GAN":
            GAN(0, cfg)


if __name__ == "__main__":
    main()
