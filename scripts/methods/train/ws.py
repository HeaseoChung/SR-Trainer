from methods.train import Trainer
from torch.nn import functional as F


class WS(Trainer):
    def __init__(self, gpu, cfg):
        super().__init__(gpu, cfg)
        self.generator = self._init_model(cfg.models.generator, cfg)
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
                average = self._valid(self.generator)
                self._print(average,step=i)
                self._save_model("g", i, self.generator, self.g_optim, average)

    def _train(self, iter):
        self.generator.train()

        lr, hr = next(self.train_dataloader)
        lr = lr.to(self.gpu)
        hr = hr.to(self.gpu)

        preds = self.generator(lr)

        losses = {}
        for l in self.loss_lists.keys():
            losses[l] = self.loss_lists[l](preds, hr)

        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        self.generator.zero_grad()
        total_loss.backward()
        self.g_optim.step()
        self.g_scheduler.step()

        if iter % self.save_log_every == 0 and self.gpu == 0:
            self._print(losses,step=iter)

        if iter % self.save_img_every == 0 and self.gpu == 0:
            lr = F.interpolate(lr, scale_factor=self.scale, mode="nearest")
            self._visualize(iter, [hr, lr, preds])
