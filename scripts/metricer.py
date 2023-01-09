import hydra
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

from valid import Valider
from torch.nn import functional as F


class Metricer(Valider):
    def __init__(self, cfg, gpu):
        super().__init__(gpu, cfg)
        self.scale = cfg.models.generator.scale
        self.generator = self._init_model(cfg.models.generator)
        self.generator = self._load_state_dict(
            cfg.models.generator.path, self.generator
        )

        self._init_dataset(cfg)
        self._init_metrics(cfg)
        self._run()

    def _run(self):
        self.generator.eval()
        scores = {}

        for m in self.metrics:
            scores[m.name] = []

        for i, (lr, hr) in enumerate(self.valid_dataloader):
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


@hydra.main(config_path="../configs", config_name="valid.yaml")
def main(cfg):
    cudnn.benchmark = True
    cudnn.deterministic = True
    Metricer(cfg, 0)


if __name__ == "__main__":
    main()
