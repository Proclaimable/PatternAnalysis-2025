"""
modules.py

Description:
    This module contains various utility functions and helper classes
    used for the [Project Name/Description]. It includes functions for
    [list of core functionalities, e.g., data preprocessing, model training,
    etc.]. The functions are modular and can be easily imported into other
    parts of the project.

Usage:
    - Function 1: [Brief description of how to use]
    - Function 2: [Brief description of how to use]
    - Etc.

Notes:
    - [Any special notes about the module, e.g., performance considerations]
"""
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import umap

# ----------
# Constants
# ---------
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 3
COLOR_MODE = "grayscale"

class Unet(tf.keras.Model):

    # numbers inspired from unet segmentation code on google colab
    def __init__(self, in_channels=3, out_channels=1, dropout_p=0.2, leaky_relu_alpha=0.2):
        
        super(Unet, self).__init__()
        self.upsample = layers.Conv2DTranspose(128, 3, strides=2, padding="same")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_p = dropout_p
        self.leaky_relu_alpha = leaky_relu_alpha

    def cnn_Block(self, x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = layers.Dropout(self.dropout_p)(x)
        return x

    def get_model(self):
        # Encoder
        # Architecture con2D -> LeakyReLU -> Dropout kernal of 3*3
        encoder_inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], self.in_channels), batch_size=BATCH_SIZE)
        c1 = self.cnn_Block(encoder_inputs, 64)
        e1 = layers.MaxPool2D(2)(c1)

        c2 = self.cnn_Block(e1, 128)
        e2 = layers.MaxPool2D(2)(c2)

        c3 = self.cnn_Block(e2, 256)
        e3 = layers.MaxPool2D(2)(c3)

        c4 = self.cnn_Block(e3, 512)
        e4 = layers.MaxPool2D(2)(c4)

        bottleneck = self.cnn_Block(e4, 1024)

        self.encoder = keras.Model(inputs=encoder_inputs, outputs=bottleneck, name="encoder")
        self.encoder.summary()


        # Decoder
        u4 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(bottleneck)
        s4 = tf.concat([u4, c4], axis=-1)
        d4 = self.cnn_Block(s4, 512)
        d4 = layers.Dropout(self.dropout_p)(d4)

        d3 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(d4)
        s3 = tf.concat([d3, c3], axis=-1)
        d3 = self.cnn_Block(s3, 256)
        d3 = layers.Dropout(self.dropout_p)(d3)

        d2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(d3)
        s2 = tf.concat([d2, c2], axis=-1)
        d2 = self.cnn_Block(s2, 128)
        d2 = layers.Dropout(self.dropout_p)(d2)

        d1 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(d2)
        s1 = tf.concat([d1, c1], axis=-1)
        d1 = self.cnn_Block(s1, 64)
        d1 = layers.Dropout(self.dropout_p)(d1)

        d1 = layers.Conv2D(self.out_channels, 1, padding="same")(d1)
        d1 = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(d1)
        self.decoder = keras.Model(inputs=e4, outputs=d1, name="decoder", final_activation=None)
        self.decoder.summary()

        model = keras.Model(inputs=encoder_inputs, outputs=d1, name="unet")

        return model
    


class DiceLoss(tf.keras.losses.Loss):
    """
    Dice Loss = 1 - Dice Coefficient
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Sigmoid output from model [B, H, W] (values between 0-1)
            targets: Binary ground truth [B, H, W] (values 0 or 1)
        """
        # Flatten tensors using reshape to handle non-contiguous memory layout
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1).float()

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        # Return Dice Loss (1 - Dice Coefficient)
        return 1 - dice_coeff





