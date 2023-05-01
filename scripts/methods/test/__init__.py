import os
import torch

from methods import Base
from archs import define_model


class Tester(Base):
    def __init__(self, cfg, gpu):
        self.gpu = gpu
        self.scale = cfg.models.generator.scale
        self.image_path = cfg.test.common.image_path
        self.save_path = cfg.test.common.save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.generator = self._init_model(cfg.models.generator)
        self.generator = self._load_state_dict(
            cfg.models.generator.path, self.generator
        )

    def _init_model(self, cfg):
        return define_model(cfg, self.gpu).eval()

    def _load_state_dict(self, path, model):
        if path:
            ckpt = torch.load(path, map_location=lambda storage, loc: storage)
            if len(ckpt) == 3:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
        return model
