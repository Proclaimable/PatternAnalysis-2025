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
from xml.parsers.expat import model
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.losses import BinaryCrossentropy
import os


# ----------
# Constants
# ---------
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 5
COLOR_MODE = "rgb"

class Unet(keras.Model):


    # numbers inspired from unet segmentation code on google colab
    def __init__(self, in_channels=3, out_channels=1, dropout_p=0.1, leaky_relu_alpha=0.2, base=64):
        
        super(Unet, self).__init__()
        self.upsample = layers.UpSampling2D(size=2, interpolation="bilinear")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_p = dropout_p
        self.leaky_relu_alpha = leaky_relu_alpha
        self.base = base


    def cnn_Block(self, x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = layers.Dropout(self.dropout_p)(x)
        return x

    def get_model(self):
        # Encoder
        # Architecture con2D -> LeakyReLU -> Dropout, kernel of 3*3
        encoder_inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], self.in_channels))
        c1 = self.cnn_Block(encoder_inputs, self.base)
        e1 = layers.MaxPool2D(2)(c1)

        c2 = self.cnn_Block(e1, self.base * 2)
        e2 = layers.MaxPool2D(2)(c2)

        c3 = self.cnn_Block(e2, self.base * 4)
        e3 = layers.MaxPool2D(2)(c3)

        c4 = self.cnn_Block(e3, self.base * 8)
        e4 = layers.MaxPool2D(2)(c4)

        bottleneck = self.cnn_Block(e4, self.base * 16)

        self.encoder = keras.Model(inputs=encoder_inputs, outputs=bottleneck, name="encoder")


        # Decoder
        u4 = layers.UpSampling2D(size=2, interpolation="bilinear")(bottleneck)
        s4 = tf.concat([u4, c4], axis=-1)
        d4 = self.cnn_Block(s4, self.base * 8)


        d3 = layers.UpSampling2D(size=2, interpolation="bilinear")(d4)
        s3 = tf.concat([d3, c3], axis=-1)
        d3 = self.cnn_Block(s3, self.base * 4)


        d2 = layers.UpSampling2D(size=2, interpolation="bilinear")(d3)
        s2 = tf.concat([d2, c2], axis=-1)
        d2 = self.cnn_Block(s2, self.base * 2)


        d1 = layers.UpSampling2D(size=2, interpolation="bilinear")(d2)
        s1 = tf.concat([d1, c1], axis=-1)
        d1 = self.cnn_Block(s1, self.base)

        d1 = layers.Conv2D(self.out_channels, 1, padding="same")(d1)
        d1 = layers.Activation("sigmoid", dtype=tf.float32)(d1)

        model = keras.Model(inputs=encoder_inputs, outputs=d1, name="unet")
        model.summary()

        return model


class ShowPredictions(keras.callbacks.Callback):
    def __init__(self, dataset, n=3, visualize_every=1, outdir="photo_storage"):
        self.dataset = dataset
        self.n = n
        self.visualize_every = visualize_every
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)


    def on_epoch_end(self, epoch, logs=None):

        if self.visualize_every is None or epoch % self.visualize_every != 0:
            return
        
        for img_batch, mask_batch in self.dataset.take(self.n):
            preds = self.model.predict(img_batch, verbose=0)
            img = img_batch[0].numpy()
            mask = mask_batch[0].numpy().squeeze()
            pred = preds[0].squeeze()

            plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(img.squeeze(), cmap='gray' if img.shape[-1] == 1 else None)
            plt.title("Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='gray')
            plt.title("Mask")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(pred, cmap='jet', alpha=0.6)
            plt.title("Prediction")
            plt.axis('off')

            filename = f"epoch_{epoch+1}_sample.png"
            save_path = os.path.join(self.outdir, filename)
            plt.savefig(save_path)
            plt.close()
        



class DiceLossBCE(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, name="dice_loss"):
        super(DiceLossBCE, self).__init__(name=name)
        self.smooth = smooth
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        bce = self.bce(y_true, y_pred)
        inter = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
        denom = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
        dice = (2. * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1. - dice

        # Print both for debugging
        #tf.print(" BCE:", bce, "Dice:", tf.reduce_mean(dice_loss), summarize=5)

        return 0.5 * bce + 0.5 * tf.reduce_mean(dice_loss)
    
    def get_config(self):
        cfg = super(DiceLossBCE, self).get_config()
        cfg.update({"smooth": self.smooth})
        return cfg
    


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, name="dice_loss"):
        super(DiceLoss, self).__init__(name=name)
        self.smooth = smooth
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_coeff = (2.0 * intersection + self.smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth)
        return 1.0 - dice_coeff
    
    def get_config(self):
        cfg = super(DiceLoss, self).get_config()
        cfg.update({"smooth": self.smooth})
        return cfg



