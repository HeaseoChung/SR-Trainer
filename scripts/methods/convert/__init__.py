import os
import torch

from torch import nn
from archs import define_model
from methods import Base
from archs.Utils.utils import *


class ONNX(Base):
    def __init__(self, cfg, gpu):
        self.scale = cfg.models.generator.scale
        self.size = [
            cfg.data.batch,
            cfg.data.channel,
            cfg.data.height,
            cfg.data.width,
        ]
        self.gpu = gpu
        self.model_name = cfg.models.generator.name
        self.generator = self._init_model(cfg.models.generator)
        self.generator = self._load_state_dict(
            cfg.models.generator.path, self.generator
        )
        self._run()

    def _init_model(self, cfg):
        return define_model(cfg, self.gpu)

    def _load_state_dict(self, path, model):
        if path:
            ckpt = torch.load(
                path,
                map_location=lambda storage, loc: storage,
            )
            if len(ckpt) == 3:
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(ckpt["model"])
                else:
                    model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
        else:
            print("State dictionary: Checkpoint is not going to be used")
        return model

    def _run(self):
        self.generator.eval()
        inputs = torch.randn(
            self.size[0], self.size[1], self.size[2], self.size[3]
        ).to(self.gpu)

        out = self.generator(inputs)

        torch.onnx.export(
            self.generator,
            inputs,
            f"{self.model_name}_x{self.scale}_{self.size[0]}_{self.size[1]}_{self.size[2]}_{self.size[3]}.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )
