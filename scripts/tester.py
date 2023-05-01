import hydra
import torch

from methods.test.image import Image
from methods.test.video import Video


@hydra.main(config_path="../configs", config_name="test.yaml")
def main(cfg):
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False

    if cfg.test.common.method == "image":
        Image(cfg, 0)
    elif cfg.test.common.method == "video":
        Video(cfg, 0)
    else:
        raise ValueError("Nether image or video")


if __name__ == "__main__":
    main()
