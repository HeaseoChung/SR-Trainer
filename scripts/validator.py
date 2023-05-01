import hydra
import torch

from methods.valid import Valider


@hydra.main(config_path="../configs", config_name="valid.yaml")
def main(cfg):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    Valider(cfg, 0)


if __name__ == "__main__":
    main()
