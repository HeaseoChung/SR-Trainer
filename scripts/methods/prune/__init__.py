import os
import torch
import torch_pruning as tp

from torch import nn
from archs import define_model
from methods import Base
from archs.Utils.utils import *


class Pruner(Base):
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
        print("Before pruning:")
        print(self.generator)

        inputs = torch.randn(1, 3, 1080, 1920).cuda()

        DG = tp.DependencyGraph().build_dependency(
            self.generator, example_inputs=inputs
        )

        pruning_idxs = pruning_idxs = [2]
        pruning_group = DG.get_pruning_group(
            self.generator.conv_1, tp.prune_conv_out_channels, idxs=pruning_idxs
        )
        if DG.check_pruning_group(pruning_group):
            pruning_group.prune()

        print("After pruning:")
        print(self.generator)

        all_groups = list(DG.get_all_groups())
        print("Number of Groups: %d" % len(all_groups))
        print("The last Group:", all_groups[-1])

        # self.generator.zero_grad()  # We don't want to store gradient information

        out = self.generator(inputs)

        torch.onnx.export(
            self.generator,
            inputs,
            f"pruned_model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

        # torch.save(
        #     self.generator.state_dict(),
        #     "pruned_model.pth",
        # )  # without .state_dict
        # self.generator = torch.load('model.pth') # load the model object
