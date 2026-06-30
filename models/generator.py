import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels: list[int], latent_dim: int=200) -> None:
        super().__init__()
        self.in_channels = [latent_dim] + in_channels
        layers = []

        for idx, (in_ch, out_ch) in enumerate(zip(self.in_channels, self.in_channels[1:])):
            layers.append(
                nn.ConvTranspose3d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=4,
                    stride=1 if idx == 0 else 2,
                    padding=0 if idx == 0 else 1,
                    bias=(idx == len(self.in_channels) - 2),
                )
            )
            if not idx == len(self.in_channels) - 2:
                layers.append(nn.BatchNorm3d(out_ch))
                layers.append(nn.ReLU(inplace=True))

        self.generator_layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input.view(input.size(0), input.size(1), 1, 1, 1)
        out = self.generator_layers(out)
        return out
