import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_features=128):
        """
        U-Net model for image-to-image tasks.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            out_channels (int): Number of output channels.
            base_features (int): Number of features in the first convolutional layer.
        """
        super(UNet, self).__init__()
        pool_size = 2
        filter_size = 4
        # Encoder blocks with pooling
        self.encoder1 = self._conv_block(in_channels, base_features, filter_size)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_size, stride=2)

        self.encoder2 = self._conv_block(base_features, base_features * 2, filter_size)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_size, stride=2)
  
        self.encoder3 = self._conv_block(base_features * 2, base_features * 4, filter_size)
        self.pool3 = nn.MaxPool2d(kernel_size=pool_size, stride=2)

        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 4, base_features * 8, filter_size)

        # Decoder blocks with transposed convolutions
        self.deconv3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=pool_size, stride=pool_size)  # Transposed convolution
        self.deconv_conv3 = self._conv_block(base_features * 8, base_features * 4, filter_size)  # Convolutional block
        self.deconv2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=pool_size, stride=pool_size)  # Transposed convolution
        self.deconv_conv2 = self._conv_block(base_features * 4, base_features * 2, filter_size)  # Convolutional block
        self.deconv1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=pool_size, stride=pool_size)  # Transposed convolution
        self.deconv_conv1 = self._conv_block(base_features * 2, base_features, filter_size)  # Convolutional block

        # Final output layer
        self.final_conv = nn.Conv2d(base_features, out_channels, kernel_size=filter_size, padding='same')

    def _conv_block(self, in_channels, out_channels, filter_size):
        """
        Creates a convolutional block with two convolutional layers followed by ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: A sequential block of convolutional layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=filter_size, padding='same'),
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
        pool_enc1 = self.pool1(enc1)  # Output after pooling encoder1
        enc2 = self.encoder2(pool_enc1)  # Output after encoder2 (with pooling)
        pool_enc2 = self.pool2(enc2)  # Output after pooling encoder2
        enc3 = self.encoder3(pool_enc2)  # Output after encoder3 (with pooling)
        pool_end3 = self.pool3(enc3)  # Output after pooling encoder3

        # Bottleneck
        bottleneck = self.bottleneck(pool_end3)
        
        dec3 = self.deconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.deconv_conv3(dec3)
        
        dec2 = self.deconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.deconv_conv2(dec2)
        
        dec1 = self.deconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.deconv_conv1(dec1)
        final = self.final_conv(dec1)
        # Final output
        return torch.tanh(final)