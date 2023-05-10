from torch import nn


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        num_in_ch = cfg.n_colors
        num_feat = cfg.num_feat

        self.conv1 = nn.Conv2d(
            num_in_ch, num_feat, kernel_size=9, padding=9 // 2
        )
        self.conv2 = nn.Conv2d(
            num_feat, num_feat // 2, kernel_size=5, padding=5 // 2
        )
        self.conv3 = nn.Conv2d(
            num_feat // 2, num_in_ch, kernel_size=5, padding=5 // 2
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
