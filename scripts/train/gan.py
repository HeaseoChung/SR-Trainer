from train import Trainer


class GAN(Trainer):
    def __init__(self, gpu, cfg):
        super().__init__(gpu, cfg)
        self.generator = self._init_model(cfg.models.generator)
        self.discriminator = self._init_model(cfg.models.discriminator)

        if cfg.train.ddp.distributed:
            self.generator = self._init_distributed_data_parallel(
                cfg, self.generator
            )
            self.discriminator = self._init_distributed_data_parallel(
                cfg, self.discriminator
            )

        self.g_optim = self._init_optim(cfg, self.generator)
        self.d_optim = self._init_optim(cfg, self.discriminator)

        self.generator, self.g_optim = self._load_state_dict(
            cfg.models.generator.path, self.generator, self.g_optim
        )

        self.discriminator, self.d_optim = self._load_state_dict(
            cfg.models.discriminator.path, self.discriminator, self.d_optim
        )

        if (
            cfg.models.generator.path == ""
            or cfg.models.discriminator.path == ""
        ):
            self.start_iters = 0

        self.g_scheduler = self._init_scheduler(cfg, self.g_optim)
        self.d_scheduler = self._init_scheduler(cfg, self.d_optim)

        self._init_loss(cfg, gpu)
        self._init_dataset(cfg)
        self._init_metrics(cfg)
        self.run()

    def run(self):
        for i in range(self.start_iters, self.end_iters):
            self.train(i)

            if i % self.save_model_every == 0 and self.gpu == 0:
                average = self._test(self.generator)
                self._save_model("g", i, self.generator, self.g_optim, average)
                self._save_model(
                    "d", i, self.discriminator, self.d_optim, average
                )

    def train(self, iter):
        self.generator.train()
        self.discriminator.train()

        def requires_grad(model, flag=True):
            for p in model.parameters():
                p.requires_grad = flag

        lr, hr = next(self.train_dataloader)
        lr = lr.to(self.gpu)
        hr = hr.to(self.gpu)

        requires_grad(self.generator, False)
        requires_grad(self.discriminator, True)

        d_loss = 0.0

        preds = self.generator(lr)
        real_pred = self.discriminator(hr)
        d_loss_real = self.loss_lists["GANLoss"](real_pred, True)

        fake_pred = self.discriminator(preds)
        d_loss_fake = self.loss_lists["GANLoss"](fake_pred, False)

        d_loss = (d_loss_real + d_loss_fake) / 2

        self.discriminator.zero_grad()
        d_loss.backward()
        self.d_optim.step()

        requires_grad(self.generator, True)
        requires_grad(self.discriminator, False)

        preds = self.generator(lr)
        fake_pred = self.discriminator(preds)

        g_loss = 0.0
        for t_loss in self.loss_lists.keys():
            if t_loss == "GANLoss":
                g_loss += self.loss_lists[t_loss](fake_pred, True)
            else:
                g_loss += self.loss_lists[t_loss](preds, hr)

        self.generator.zero_grad()
        g_loss.backward()
        self.g_optim.step()

        self.g_scheduler.step()
        self.d_scheduler.step()

        if iter % self.save_img_every == 0 and self.gpu == 0:
            self._visualize(hr, lr, preds)
