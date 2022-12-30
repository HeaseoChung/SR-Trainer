from metric.metrics import *


def define_metrics(cfg):
    metrics = []
    for m in cfg.train.metrics.types:
        if m == "psnr":
            metrics.append(PSNR())
        elif m == "ssim":
            metrics.append(SSIM())
        elif m == "lpips":
            metrics.append(LPIPS())
        elif m == "erqa":
            metrics.append(ERQA())

    print(f"Metrics: {cfg.train.metrics.types} is going to be used")
    return metrics