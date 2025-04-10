import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_features=64):
        """
        U-Net model for image-to-image tasks.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            out_channels (int): Number of output channels.
            base_features (int): Number of features in the first convolutional layer.
        """
        super(UNet, self).__init__()

        # Encoder blocks with pooling
        self.encoder1 = nn.Sequential(
            self._conv_block(in_channels, base_features),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder2 = nn.Sequential(
            self._conv_block(base_features, base_features * 2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder3 = nn.Sequential(
            self._conv_block(base_features * 2, base_features * 4),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder4 = nn.Sequential(
            self._conv_block(base_features * 4, base_features * 8),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)

        # Decoder blocks
        self.decoder4 = self._conv_block(base_features * 16, base_features * 8)
        self.decoder3 = self._conv_block(base_features * 8, base_features * 4)
        self.decoder2 = self._conv_block(base_features * 4, base_features * 2)
        self.decoder1 = self._conv_block(base_features * 2, base_features)

        # Final output layer
        self.final_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        """
        Creates a convolutional block with two convolutional layers followed by ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: A sequential block of convolutional layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Encoder
        enc1 = self.encoder1(x)  # Output after encoder1 (with pooling)
        enc2 = self.encoder2(enc1)  # Output after encoder2 (with pooling)
        enc3 = self.encoder3(enc2)  # Output after encoder3 (with pooling)
        enc4 = self.encoder4(enc3)  # Output after encoder4 (with pooling)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder
        dec4 = self._upsample_and_concat(bottleneck, enc4)
        dec4 = self.decoder4(dec4)

        dec3 = self._upsample_and_concat(dec4, enc3)
        dec3 = self.decoder3(dec3)

        dec2 = self._upsample_and_concat(dec3, enc2)
        dec2 = self.decoder2(dec2)

        dec1 = self._upsample_and_concat(dec2, enc1)
        dec1 = self.decoder1(dec1)

        # Final output
        return self.final_conv(dec1)

    def _upsample_and_concat(self, x1, x2):
        """
        Upsamples x1 to the size of x2 and concatenates them along the channel dimension.

        Args:
            x1 (torch.Tensor): Tensor to be upsampled.
            x2 (torch.Tensor): Tensor to be concatenated with.

        Returns:
            torch.Tensor: Concatenated tensor.
        """
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([x1, x2], dim=1)