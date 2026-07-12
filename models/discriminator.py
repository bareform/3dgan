import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, in_channels: list[int], use_spectral_norm: bool=False) -> None:
        super().__init__()
        self.in_channels = in_channels + [1]
        layers = []
        for idx, (in_ch, out_ch) in enumerate(zip(self.in_channels, self.in_channels[1:])):
            layers.append(
                spectral_norm(nn.Conv3d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=4,
                    stride=1 if (idx == len(self.in_channels) - 2) else 2,
                    padding=0 if (idx == len(self.in_channels) - 2) else 1,
                    bias=False,
                )) if use_spectral_norm else nn.Conv3d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=4,
                    stride=1 if (idx == len(self.in_channels) - 2) else 2,
                    padding=0 if (idx == len(self.in_channels) - 2) else 1,
                    bias=False,
                )
            )
            if not (idx == len(self.in_channels) - 2):
                if idx != 0 and not use_spectral_norm:
                    layers.append(nn.BatchNorm3d(out_ch))
                layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.discriminator_layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.discriminator_layers(input)
        out = out.view(out.size(0), -1)
        return out
