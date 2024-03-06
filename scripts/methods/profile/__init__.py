import os
import torch

from torch import nn
from archs import define_model
from methods import Base
from archs.Utils.utils import *


class Profiler(Base):
    def __init__(self, cfg, gpu):
        self.scale = cfg.models.generator.scale
        self.size = [cfg.data.batch, cfg.data.channel, cfg.data.height, cfg.data.width]
        self.gpu = gpu
        self.model_name = cfg.models.generator.name
        self.generator = self._init_model(cfg.models.generator)
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
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        self.generator.eval()
        inputs = torch.randn(self.size[0], self.size[1], self.size[2], self.size[3]).to(
            self.gpu
        )
        input_dim = (self.size[1], self.size[2], self.size[3])

        # GPU Warming up
        for _ in range(10):
            with torch.no_grad():
                self.generator(inputs)

        with torch.no_grad():
            start.record()
            self.generator(inputs)
            end.record()
            torch.cuda.synchronize()

        print(
            "------> Average runtime of {} is : {:.6f} ms".format(
                self.model_name, start.elapsed_time(end)
            )
        )

        flops = get_model_flops(self.generator, input_dim, False)
        flops = flops / 10**9
        print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        activations, num_conv = get_model_activation(self.generator, input_dim)
        activations = activations / 10**6
        print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
        print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

        num_parameters = sum(map(lambda x: x.numel(), self.generator.parameters()))
        num_parameters = num_parameters / 10**6
        print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))

        max_mem = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024**2
        print("{:>16s} : {:<.3f} [M]".format("Max Memory", max_mem))
