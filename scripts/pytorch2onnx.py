import hydra
import torch

from methods.convert import ONNX


@hydra.main(config_path="../configs", config_name="pytorch2onnx.yaml")
def main(cfg):
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    ONNX(cfg, 0)


if __name__ == "__main__":
    main()
