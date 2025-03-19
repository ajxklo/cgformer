from collections import OrderedDict

from .model.layers import *
import numpy as np
import torch
import torch.nn as nn

import torchvision.models as models


# Code taken from https://github.com/nianticlabs/monodepth2
#
# Godard, Clément, et al.
# "Digging into self-supervised monocular depth estimation."
# Proceedings of the IEEE/CVF international conference on computer vision.
# 2019.


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if pretrained:
            self.encoder = resnets[num_layers](pretrained=True)
        else:
            self.encoder = resnets[num_layers]()

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class Decoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(Decoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x +=[input_features[i-1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                if i == 0:
                    self.outputs[("disp", i)] = self.convs[("dispconv", i)](x)
                else:
                    self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs


class RID(nn.Module):
    def __init__(
            self,
            resnet_layers=18,
            cp_location=None,
            d_out=128,
            scales=range(4)
    ):
        super().__init__()

        self.encoder = ResnetEncoder(resnet_layers, True, 1)
        self.num_ch_enc = self.encoder.num_ch_enc

        self.upsample_mode = 'nearest'
        self.d_out = d_out
        self.scales = scales

        # decoder
        self.decoder = Decoder(num_ch_enc=self.num_ch_enc, num_output_channels=1)
        self.num_ch_dec = self.decoder.num_ch_dec

        self.latent_size = self.d_out

        if cp_location is not None:
            cp = torch.load(cp_location)
            self.load_state_dict(cp["model"])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x: image (B, C, H, W)
        """
        x = torch.cat([x * .5 + .5], dim=1)
        image_features = self.encoder(x)
        res_inv_depth = self.decoder(image_features)[("disp", 0)]

        return res_inv_depth
