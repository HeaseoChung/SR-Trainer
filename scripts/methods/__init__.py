import torch
from abc import ABC, abstractmethod


class Base(ABC):
    def __init__(self, gpu, cfg) -> None:
        super().__init__()
        ### GPU Device setting
        self.gpu = gpu
        self.ngpus_per_node = torch.cuda.device_count()
        self.scale = cfg.models.generator.scale

        ### Model setting
        self.generator = None
        self.discriminator = None

    @abstractmethod
    def _init_model(self, cfg):
        pass

    @abstractmethod
    def _load_state_dict(self, path, model, optim):
        pass
