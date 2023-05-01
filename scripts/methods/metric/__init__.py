from methods.metric.metrics import *


def define_metrics(cfg):
    metrics = {}
    for m in cfg.metrics.types:
        if m == "psnr":
            metrics[m] = calculate_psnr
        elif m == "ssim":
            metrics[m] = calculate_ssim

    print(f"Metrics: {cfg.metrics.types} is going to be used")
    return metrics
