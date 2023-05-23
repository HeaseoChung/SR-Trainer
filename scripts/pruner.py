import hydra
import torch

from methods.prune import Pruner


@hydra.main(config_path="../configs", config_name="prune.yaml")
def main(cfg):
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    Pruner(cfg, 0)


if __name__ == "__main__":
    main()
