import torch
import torch.nn as nn


class SimpleBEVModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleBEVModel, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 3D convolution layers with reduced channels to save memory
        self.conv3d_1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)

        # BEV head (2D)
        # self.pooling = nn.MaxPool3d(kernel_size=(1, 1, 32), stride=(1, 1, 32))
        self.bev_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, lss_volume):
        # lss_volume shape: (batch_size, in_channels, D, H, W)

        # Apply 3D convolutions to extract 3D features (with reduced channels)
        x = self.conv3d_1(lss_volume)
        x = self.conv3d_2(x)

        # Convert 3D features to 2D by pooling along the z-axis
        # x = self.pooling(x).squeeze(-1)
        x = x.sum(dim=4,keepdim=True).squeeze(-1)
        # Apply BEV head
        bev_pred = self.bev_head(x)
        return bev_pred
