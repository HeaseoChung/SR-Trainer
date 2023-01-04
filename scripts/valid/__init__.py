import os
import torch
import torchvision.utils as vutils

from torch import nn
from torch.utils.data.dataloader import DataLoader
from data import define_dataset
from archs import define_model
from metric import define_metrics


class Valider:
    def __init__(self, gpu, cfg):
        ### GPU Device setting
        self.gpu = gpu
        self.ngpus_per_node = torch.cuda.device_count()
        self.scale = cfg.models.generator.scale

        ### Common setting
        self.save_path = cfg.valid.common.ckpt_dir
        os.makedirs(self.save_path, exist_ok=True)
        self.seed = cfg.valid.common.seed

        ### Model setting
        self.generator = None
        self.discriminator = None

        self.valid_dataloader = None

        ### Metrics setting
        self.metrics = None

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
                self.start_iters = ckpt["iteration"] + 1
            else:
                model.load_state_dict(ckpt)
        else:
            self.start_iters = 0
            print("State dictionary: Checkpoint is not going to be used")
        return model

    def _init_dataset(self, cfg):

        cfg.valid.dataset.common.sf = self.scale
        valid_dataset = define_dataset(
            cfg.valid.dataset.common, cfg.valid.dataset.valid
        )

        self.valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=cfg.valid.dataset.valid.batch_size,
            shuffle=None,
            num_workers=cfg.valid.dataset.valid.num_workers,
            pin_memory=True,
            sampler=None,
            drop_last=True,
        )

    def _init_metrics(self, cfg):
        self.metrics = define_metrics(cfg.valid)

    def _visualize(self, i, img1):
        vutils.save_image(img1, os.path.join(self.save_path, f"preds_{i}.png"))
