from train import Trainer
from torch.nn import functional as F


class Net(Trainer):
    def __init__(self, gpu, cfg):
        super().__init__(gpu, cfg)
        self.generator = self._init_model(cfg.models.generator)
        if cfg.train.ddp.distributed:
            self.generator = self._init_distributed_data_parallel(
                cfg, self.generator
            )
        self.g_optim = self._init_optim(cfg, self.generator)
        self.generator, self.g_optim = self._load_state_dict(
            cfg.models.generator.path, self.generator, self.g_optim
        )
        self.g_scheduler = self._init_scheduler(cfg, self.g_optim)

        self._init_loss(cfg, gpu)
        self._init_dataset(cfg)
        self._init_metrics(cfg)
        self._run()

    def _run(self):
        for i in range(self.start_iters, self.end_iters):
            self._train(i)

            if i % self.save_model_every == 0 and self.gpu == 0:
                average = self._test(self.generator)
                print(average)
                self._save_model("s", i, self.generator, self.g_optim, average)

    def _train(self, iter):
        self.generator.train()

        lr, hr = next(self.train_dataloader)
        lr = lr.to(self.gpu)
        hr = hr.to(self.gpu)

        preds = self.generator(lr)

        loss = 0
        for t_loss in self.loss_lists.keys():
            loss += self.loss_lists[t_loss](preds, hr)

        self.generator.zero_grad()
        loss.backward()
        self.g_optim.step()
        self.g_scheduler.step()

        if iter % self.save_img_every == 0 and self.gpu == 0:
            lr = F.interpolate(lr, scale_factor=self.scale, mode="nearest")
            self._visualize(hr, lr, preds)
