import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            #! input shape mishe : Batch Size * channel_image * 64 * 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), #! 32 * 32 mishe
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), #! 16 * 16
            self._block(features_d*2, features_d*4, 4, 2, 1), #! 8 * 8
            self._block(features_d*4, features_d*8, 4, 2, 1), #! 4 * 4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0), #! 1 * 1 inja moshakhas mishe ke fake hast ya real
            nn.Sigmoid()
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )


    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):

    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()