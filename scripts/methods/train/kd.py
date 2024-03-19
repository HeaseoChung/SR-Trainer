from methods.train import Trainer
from torch.nn import functional as F


class KD(Trainer):
    def __init__(self, gpu, cfg):
        super().__init__(gpu, cfg)
        self.teacher = self._init_model(cfg.models.teacher, cfg)
        self.student = self._init_model(cfg.models.student, cfg)

        self.t_optim = self._init_optim(cfg, self.teacher)
        self.s_optim = self._init_optim(cfg, self.student)

        self.teacher, self.t_optim = self._load_state_dict(
            cfg.models.teacher.path, self.teacher, self.t_optim
        )
        self.student, self.s_optim = self._load_state_dict(
            cfg.models.student.path, self.student, self.s_optim
        )

        self.start_iters = 0

        self.t_scheduler = self._init_scheduler(cfg, self.t_optim)
        self.s_scheduler = self._init_scheduler(cfg, self.s_optim)

        self._init_loss(cfg, gpu)
        self._init_dataset(cfg)
        self._init_metrics(cfg)
        self._run()

    def _run(self):
        for i in range(self.start_iters, self.end_iters):
            self._train(i)

            if i % self.save_model_every == 0 and self.gpu == 0:
                average = self._valid(self.student)
                self._print(average,step=i)
                self._save_model("g", i, self.student, self.s_optim, average)

    def _train(self, iter):
        def requires_grad(model, flag=True):
            for p in model.parameters():
                p.requires_grad = flag

        lr, hr = next(self.train_dataloader)
        lr = lr.to(self.gpu)
        hr = hr.to(self.gpu)

        """Teacher"""
        self.teacher.train()
        requires_grad(self.teacher, False)
        t_preds = self.teacher(lr)

        """Student"""
        self.student.train()
        s_preds = self.student(lr)

        losses = {}
        for l in self.loss_lists.keys():
            if l == "Wavelet":
                losses[l] = self.loss_lists[l](s_preds, t_preds)
            else:
                losses[l] = self.loss_lists[l](s_preds, hr)

        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss
        self.student.zero_grad()
        total_loss.backward()
        self.s_optim.step()
        self.s_scheduler.step()

        if iter % self.save_log_every == 0 and self.gpu == 0:
            self._print(losses,step=iter)

        if iter % self.save_img_every == 0 and self.gpu == 0:
            self._visualize(iter, [hr, t_preds, s_preds])
