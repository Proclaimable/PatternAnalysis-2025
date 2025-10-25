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
LEAKY_RELU_ALPHA = 0.2

class Unet(tf.keras.Model):

    # numbers inspired from unet segmentation code on google colab
    def __init__(self, in_channels=3, out_channels=1, dropout_p=0.2):
        
        super(Unet, self).__init__()
        self.upsample = layers.Conv2DTranspose(128, 3, strides=2, padding="same")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_p = dropout_p
    

    def forward(self,x):
        # Encoder
        # Architecture con2D -> LeakyReLU -> Dropout kernal of 3*3
        encoder_inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], self.in_channels), batch_size=BATCH_SIZE)
        e1 = layers.Conv2D(32, 3, strides=2, padding="same")(encoder_inputs)
        e1 = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(e1)
        e1 = layers.Dropout(self.dropout_p)(e1)
        e1 = layers.MaxPool2D(2)(e1)

        e2 = layers.Conv2D(64, 3, strides=2, padding="same")(e1)
        e2 = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(e2)
        e2 = layers.Dropout(self.dropout_p)(e2)
        e2 = layers.MaxPool2D(2)(e2)

        e3 = layers.Conv2D(128, 3, strides=2, padding="same")(e2)
        e3 = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(e3)
        e3 = layers.Dropout(self.dropout_p)(e3)

        e4 = layers.Conv2D(256, 3, strides=2, padding="same")(e3)
        e4 = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(e4)
        e4 = layers.Dropout(self.dropout_p)(e4)
        e4 = layers.AvgPool2D(2)(e4)


        # Decoder
        d4 = layers.UpSampling2D(interpolation="bilinear", size=2)(e4)
        d4 = layers.Conv2D(128, 3, padding="same")(tf.concat([d4, e3], 1))
        d4 = layers.Dropout(self.dropout_p)(d4)

        d3 = layers.UpSampling2D(interpolation="bilinear", size=2)(d4)
        d3 = layers.Conv2D(64, 3, padding="same")(tf.concat([d3, e2], 1))
        d3 = layers.Dropout(self.dropout_p)(d3)

        d2 = layers.UpSampling2D(interpolation="bilinear", size=2)(d3)
        d2 = layers.Conv2D(32, 3, padding="same")(tf.concat([d2, e1], 1))
        d2 = layers.Dropout(self.dropout_p)(d2)


        d1 = layers.Conv2D(self.out_channels, 1, padding="same")(d2) 

        return d1
    
    







