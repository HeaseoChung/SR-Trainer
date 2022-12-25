import torch


def define_LR_scheduler(cfg, optimizer, start_iter):
    lr_scheduler = None
    lr_scheduler_type = cfg.train.scheduler.type

    if lr_scheduler_type == "MultiStepLR":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.train.scheduler.MultiStepLR.g_milestones,
            cfg.train.scheduler.MultiStepLR.g_gamma,
            last_epoch=start_iter if start_iter > 0 else -1,
        )
    elif lr_scheduler_type == "CosineAnnealingWarmRestarts":
        # TODO ADD CosineAnnealingWarmRestarts
        pass

    assert lr_scheduler != None, "The selected lr_scheduler is out of scope"
    print(f"Learning rate scheduler: {lr_scheduler_type} is going to be used")
    return lr_scheduler
