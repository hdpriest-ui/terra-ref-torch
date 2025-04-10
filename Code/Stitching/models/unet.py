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
        
        # Encoder blocks with pooling
        self.encoder1 = self._conv_block(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._conv_block(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
  
        self.encoder3 = self._conv_block(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._conv_block(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # # Encoder blocks with pooling
        # self.encoder1 = nn.Sequential(
        #     self._conv_block(in_channels, base_features),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.encoder2 = nn.Sequential(
        #     self._conv_block(base_features, base_features * 2),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.encoder3 = nn.Sequential(
        #     self._conv_block(base_features * 2, base_features * 4),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.encoder4 = nn.Sequential(
        #     self._conv_block(base_features * 4, base_features * 8),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )

        # Bottleneck
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)

        # Decoder blocks with transposed convolutions
        self.deconv4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)  # Transposed convolution
        self.deconv_conv4 = self._conv_block(base_features * 16, base_features * 8)  # Convolutional block
        self.deconv3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)  # Transposed convolution
        self.deconv_conv3 = self._conv_block(base_features * 8, base_features * 4)  # Convolutional block
        self.deconv2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)  # Transposed convolution
        self.deconv_conv2 = self._conv_block(base_features * 4, base_features * 2)  # Convolutional block
        self.deconv1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)  # Transposed convolution
        self.deconv_conv1 = self._conv_block(base_features * 2, base_features)  # Convolutional block
        # self.decoder4 = self._decoder_block(base_features * 16, base_features * 8)
        # self.decoder3 = self._decoder_block(base_features * 8, base_features * 4)
        # self.decoder2 = self._decoder_block(base_features * 4, base_features * 2)
        # self.decoder1 = self._decoder_block(base_features * 2, base_features)

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
        
    def _decoder_block(self, in_channels, out_channels):
        """
        Creates a decoder block with a transposed convolutional layer for upsampling
        followed by a convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: A sequential block with transposed convolution and convolutional layers.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),  # Transposed convolution
            self._conv_block(out_channels, out_channels)  # Convolutional block
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
        enc4 = self.encoder4(pool_end3)  # Output after encoder4 (with pooling)
        pool_enc4 = self.pool4(enc4)  # Output after pooling encoder4

        # Bottleneck
        bottleneck = self.bottleneck(pool_enc4)

        # Decoder
        # dec4 = self.decoder4(bottleneck)  # Upsample and process
        # dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection

        # dec3 = self.decoder3(dec4)
        # dec3 = torch.cat([dec3, enc3], dim=1)

        # dec2 = self.decoder2(dec3)
        # dec2 = torch.cat([dec2, enc2], dim=1)

        # dec1 = self.decoder1(dec2)
        # dec1 = torch.cat([dec1, enc1], dim=1)
        dec4 = self.deconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.deconv_conv4(dec4)
        
        dec3 = self.deconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.deconv_conv3(dec3)
        
        dec2 = self.deconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.deconv_conv2(dec2)
        
        dec1 = self.deconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.deconv_conv1(dec1)
        
        # Final output
        return self.final_conv(dec1)