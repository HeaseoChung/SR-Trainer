import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, gan):
        super(GANLoss, self).__init__()
        self.loss_weight = gan.loss_weight
        self.register_buffer("real_label", torch.tensor(gan.real_label_val))
        self.register_buffer("fake_label", torch.tensor(gan.fake_label_val))

        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss_weight * self.loss(input, target_tensor)
