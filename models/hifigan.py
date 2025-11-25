import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels, kernel, dilations):
        super().__init__()

        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel,
                dilation=d,
                padding=(d * (kernel - 1)) // 2
            )
            for d in dilations
        ])

        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel,
                dilation=1,
                padding=(kernel - 1) // 2
            )
            for _ in dilations
        ])

    def forward(self, x):
        out = x
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(out, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)

            # ensure equal lengths
            min_len = min(out.size(-1), xt.size(-1))
            out = out[..., :min_len]
            xt = xt[..., :min_len]

            out = out + xt

        return out


class HiFiGANGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        channels = cfg["channels"]

        self.conv_in = nn.Conv1d(cfg["num_mels"], channels, kernel_size=7, padding=3)

        self.ups = nn.ModuleList()
        for rate, k in zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"]):
            pad = (k - rate) // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    channels,
                    channels,
                    kernel_size=k,
                    stride=rate,
                    padding=pad
                )
            )

        self.resblocks = nn.ModuleList([
            ResBlock(channels, k, d)
            for k, d in zip(cfg["resblock_kernel_size"], cfg["resblock_dilation_sizes"])
        ])

        self.conv_out = nn.Conv1d(channels, 1, kernel_size=7, padding=3)

    def forward(self, mel):
        x = self.conv_in(mel)

        for up in self.ups:
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            # apply all ResBlocks then average
            rb_sum = 0
            for rb in self.resblocks:
                rb_sum = rb_sum + rb(x)
            x = rb_sum / len(self.resblocks)

        x = F.leaky_relu(x, 0.1)
        x = torch.tanh(self.conv_out(x))
        return x
