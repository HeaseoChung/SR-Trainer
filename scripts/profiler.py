import hydra
import torch

from methods.profile import Profiler


@hydra.main(config_path="../configs", config_name="profile.yaml")
def main(cfg):
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    Profiler(cfg, 0)


if __name__ == "__main__":
    main()
