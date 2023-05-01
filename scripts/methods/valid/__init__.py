import os
import torch
import torchvision.utils as vutils
import pandas as pd

from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from data import define_dataset
from archs import define_model
from metric import define_metrics
from methods import Base


class Valider(Base):
    def __init__(self, gpu, cfg):
        super().__init__(gpu, cfg)
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

        self._init_model(cfg)
        self._init_dataset(cfg)
        self._init_metrics(cfg)
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

        self.dataloader = DataLoader(
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

    def _visualize(self, i, img1, img2, img3):
        results = torch.cat(
            (
                img1.detach(),
                img2.detach(),
                img3.detach(),
            ),
            3,
        )
        vutils.save_image(
            results, os.path.join(self.save_path, f"compare_{i}.png")
        )

    def _run(self):
        self.generator.eval()
        scores = {}

        for m in self.metrics:
            scores[m.name] = []

        for i, (lr, hr) in enumerate(self.dataloader):
            lr = lr.to(self.gpu)
            hr = hr.to(self.gpu)

            with torch.no_grad():
                preds = self.generator(lr)

            for m in self.metrics:
                scores[m.name].append(m(preds, hr).item())

            lr = F.interpolate(lr, scale_factor=self.scale, mode="nearest")
            self._visualize(i, hr, lr, preds)

        df = pd.DataFrame.from_dict(scores, orient="columns")
        df.to_csv("Quantitative_Score.csv")
