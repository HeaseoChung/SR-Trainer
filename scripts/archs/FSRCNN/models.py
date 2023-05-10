from torch import nn


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        scale = cfg.scale
        num_in_ch = cfg.n_colors
        num_feat = cfg.num_feat
        num_sfeat = cfg.num_sfeat
        num_blocks = cfg.num_blocks

        self.first_part = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=5, padding=5 // 2),
            nn.PReLU(num_feat),
        )
        self.mid_part = [
            nn.Conv2d(num_feat, num_sfeat, kernel_size=1),
            nn.PReLU(num_sfeat),
        ]
        for _ in range(num_blocks):
            self.mid_part.extend(
                [
                    nn.Conv2d(
                        num_sfeat, num_sfeat, kernel_size=3, padding=3 // 2
                    ),
                    nn.PReLU(num_sfeat),
                ]
            )
        self.mid_part.extend(
            [nn.Conv2d(num_sfeat, num_feat, kernel_size=1), nn.PReLU(num_feat)]
        )
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(
            num_feat,
            num_in_ch,
            kernel_size=9,
            stride=scale,
            padding=9 // 2,
            output_padding=scale - 1,
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
