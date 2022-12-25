import torch


def define_optim(cfg, model):
    optimizer = None
    optim_type = cfg.train.optim.type
    if cfg.train.optim.type == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.optim.Adam.lr,
            betas=(
                cfg.train.optim.Adam.betas[0],
                cfg.train.optim.Adam.betas[1],
            ),
            weight_decay=cfg.train.optim.Adam.weight_decay,
        )
    elif cfg.train.optim.type == "SGD":
        # TODO ADD SGD optim
        pass

    assert optimizer != None, "The selected optimizer is out of scope"
    print(f"Optimizer: {optim_type} is going to be used")
    return optimizer
