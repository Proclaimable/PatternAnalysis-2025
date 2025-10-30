"""
modules.py

Description:
    Utilities and model components for an improved U-Net segmentation model.
    Contains the ImprovedUnet model builder, training callback(s) for saving
    prediction visuals, and loss implementations (Dice, Dice+BCE). Intended
    to be used with tf.data pipelines and tf.keras training loops.

Usage:
    - ImprovedUnet.get_model(): returns a tf.keras.Model ready to compile().
    - ShowPredictions(dataset): callback to save example predictions each epoch.
    - DiceLoss / DiceLossBCE: use in model.compile(loss=...)

Notes:
    - Use tf.keras consistently (from tensorflow import keras) for compatibility.
    - This module assumes input batches of shape (B, H, W, C) and masks in [0,1].
"""
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import os


# ----------
# Constants
# ---------
IMAGE_SIZE = (256, 256)
COLOR_MODE = "rgb"

class ImprovedUnet(keras.Model):
    """
    Improved U-Net model class.

    This class provides modular building blocks (cnn_block, context_block,
    upsample_module, localization_module, segmentation_layer) and a get_model
    method which constructs and returns a tf.keras.Model implementing the
    improved U-Net architecture with context/residual blocks and multi-scale
    segmentation fusion.
    """

    # numbers inspired from unet segmentation code on google colab
    def __init__(self, in_channels=3, out_channels=1, dropout_p=0.1, leaky_relu_alpha=0.2, base=64):
        """
        Initialize ImprovedUnet.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            out_channels (int): Number of output channels (usually 1 for binary mask).
            dropout_p (float): Dropout probability used in blocks.
            leaky_relu_alpha (float): Negative slope for LeakyReLU.
            base (int): Base filter count; deeper layers multiply this.
        """
        super(ImprovedUnet, self).__init__()
        self.upsample = layers.UpSampling2D(size=2, interpolation="bilinear")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_p = dropout_p
        self.leaky_relu_alpha = leaky_relu_alpha
        self.base = base


    def cnn_block(self, x, filters, strides = 1):
        """
        Basic double-convolution block.

        Pattern: Conv(3x3) -> LeakyReLU -> Conv(3x3) -> LeakyReLU -> Dropout.
        Optionally applies stride on the first conv for downsampling.

        Args:
            x (Tensor): Input tensor.
            filters (int): Number of output filters.
            strides (int): Stride for the first convolution (use 2 for downsampling).

        Returns:
            Tensor: Processed tensor.
        """
        # Two convolutional layers with LeakyReLU and Dropout at the end
        # con -> relu -> con -> relu -> dropout
        x = layers.Conv2D(filters, 3, padding="same", strides=strides)(x)
        x = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = layers.Dropout(self.dropout_p)(x)
        return x

    def context_block(self, x, filters, p=0.2, alpha=0.2, dilation=2):
        """
        Context (residual + dilated conv) block.

        Adds spatial context using a dilated convolution and a residual shortcut.
        If the shortcut channel count differs from filters, a 1x1 conv aligns it.

        Args:
            x (Tensor): Input tensor.
            filters (int): Number of filters for internal convolutions.
            p (float): Dropout probability.
            alpha (float): LeakyReLU alpha.
            dilation (int): Dilation rate for the second conv.

        Returns:
            Tensor: Output tensor after residual addition.
        """
        # added context to the feature maps using dilated convolutions and residual connections as the shortcut variable
        # Con -> norm -> relu -> dropout -> conv -> norm -> relu -> residual
        shortcut = x
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=alpha)(x)
        x = layers.Dropout(p)(x)
        x = layers.Conv2D(filters, 3, padding="same", dilation_rate=dilation)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=alpha)(x)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)
        x = layers.Add()([x, shortcut])
        return x

    def upsample_module(self, x, out_filters):
        """
        Upsampling module (resize + conv + activation).

        Performs non-learnable bilinear upsampling followed by a small conv to
        refine features and project to the desired channel count.

        Args:
            x (Tensor): Input tensor.
            out_filters (int): Number of filters after the conv.

        Returns:
            Tensor: Upsampled and refined tensor.
        """
        # upsamples using bilinear interpolation
        # convolutes the upsample to create a segmentation map for adding context (decoding section)
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        x = layers.Conv2D(out_filters, 3, padding="same")(x)
        x = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        return x

    def localization_module(self, x, out_filters):
        """
        Localization module used in decoder skip-connection processing.

        Uses a 3x3 conv followed by a 1x1 conv (channel bottleneck) to extract
        spatially local features and reduce channel dimensionality.

        Args:
            x (Tensor): Input tensor.
            out_filters (int): Target filters for the 3x3 conv.

        Returns:
            Tensor: Localized feature tensor.
        """
        # Local feature extraction and context aggregation
        # Using convolutions tat a 1x1 pointwise kernal to extract local features
        x = layers.Conv2D(out_filters, 3, padding="same")(x)
        x = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        x = layers.Conv2D(out_filters // 2, 1, padding="same")(x)
        x = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(x)
        return x
    
    def segmentation_layer(self, x, out_channels=1):
        """
        Final segmentation 1x1 conv layer.

        Applies a 1x1 convolution with sigmoid activation to map features to
        segmentation logits/probabilities for each pixel.

        Args:
            x (Tensor): Input tensor.
            out_channels (int): Number of output channels (default 1).

        Returns:
            Tensor: Segmentation output tensor.
        """
        # 1x1 convolution for segmentation for the output after added upsampled weights
        x = layers.Conv2D(out_channels, 1, activation="sigmoid", padding="same")(x)
        return x


    def get_model(self):
        """
        Build and return a tf.keras.Model implementing the Improved U-Net.

        The encoder uses cnn_block + context_block stacks; the decoder upsamples,
        concatenates with encoder context features and applies localization blocks.
        Multi-scale segmentation maps are fused to produce the final output.

        Returns:
            tf.keras.Model: compiled model architecture (not compiled with optimizer/loss here).
        """
        # Encoder.
        # Repeats convolution blocks with context block using residual connections
        encoder_inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], self.in_channels))

        conv1 = self.cnn_block(encoder_inputs, self.base)
        cont1 = self.context_block(conv1, self.base)

        conv2 = self.cnn_block(cont1, self.base * 2, strides=2)
        cont2 = self.context_block(conv2, self.base * 2)

        conv3 = self.cnn_block(cont2, self.base * 4, strides=2)
        cont3 = self.context_block(conv3, self.base * 4)

        conv4 = self.cnn_block(cont3, self.base * 8, strides=2)
        cont4 = self.context_block(conv4, self.base * 8)


        # Bottle neck is the deepest layer with the most abstract features
        bottleneck = self.cnn_block(cont4, self.base * 16, strides=2)
        bottleneck = self.context_block(bottleneck, self.base * 16)

        # Decoder.
        # Upsamples and concatenates with skip connections of encoder features
        u4 = self.upsample_module(bottleneck, self.base * 8)
        cat4 = layers.Concatenate()([u4, cont4])
        loc4 = self.localization_module(cat4, self.base * 8)

        u3 = self.upsample_module(loc4, self.base * 4)
        cat3 = layers.Concatenate()([u3, cont3])
        loc3 = self.localization_module(cat3, self.base * 4)

        u2 = self.upsample_module(loc3, self.base * 2)
        cat2 = layers.Concatenate()([u2, cont2])
        loc2 = self.localization_module(cat2, self.base * 2)

        u1 = self.upsample_module(loc2, self.base)
        cat1 = layers.Concatenate()([u1, cont1])
        convout = self.cnn_block(cat1, self.base)

        # 1x1 convolution for segmentation
        segout = self.segmentation_layer(convout)


        # segment localization layers
        seg3 = self.segmentation_layer(loc3, 1)
        seg2 = self.segmentation_layer(loc2, 1)
        # upscale
        seg3_up = layers.UpSampling2D(size=4, interpolation="bilinear")(seg3)
        seg2_up = layers.UpSampling2D(size=2, interpolation="bilinear")(seg2)

        # add localization layers of 3, 2 and final layer
        fuse = layers.Concatenate()([segout, seg3_up, seg2_up])

        # convolution to produce final logits enabling activation
        final_logits = layers.Conv2D(1, 1, padding="same")(fuse)

        # sigmoid activation to have a 0-1 output 
        outputs = layers.Activation("sigmoid")(final_logits)

        model = keras.Model(inputs=encoder_inputs, outputs=outputs, name="unet")
        model.summary()

        return model



class ShowPredictions(keras.callbacks.Callback):
    """
    Callback that saves example predictions to disk at the end of selected epochs.

    The callback draws a three-panel figure (image, mask, prediction) for a
    randomly sampled example from the first batch of the provided dataset and
    saves it to the configured output directory.
    """
    def __init__(self, dataset, n=3, visualize_every=1, outdir="photo_storage_Improved"):
        """
        Initialize the ShowPredictions callback.

        Args:
            dataset (tf.data.Dataset): Dataset yielding (image_batch, mask_batch).
            n (int): Unused here but reserved for future multi-sample saving.
            visualize_every (int): Save visuals every N epochs (1 = every epoch).
            outdir (str): Directory to save images.
        """
        self.dataset = dataset
        self.n = n
        self.visualize_every = visualize_every
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)


    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.

        If the epoch index matches the configured frequency, run model.predict on
        a batch, render image/mask/prediction and save to disk.
        """
        # creates a visualization of current masks without binary thresholding
        # outputs a plot of a random mask, image and prediction into a file path
        if self.visualize_every is None or epoch % self.visualize_every != 0:
            return

        batch = next(iter(self.dataset.take(1)))
        img_batch, mask_batch = batch
        batch_size = img_batch.shape[0]
        idx = 1     #np.random.randint(0, min(20, batch_size))

        preds = self.model.predict(img_batch, verbose=0)
        img = img_batch[idx].numpy()
        mask = mask_batch[idx].numpy().squeeze()
        pred = preds[idx].squeeze()

        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(img.squeeze(), cmap='gray' if img.shape[-1] == 1 else None)
        plt.title(f"Image {idx}")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap='jet', alpha=0.6)
        plt.title("Prediction")
        plt.axis('off')

        os.makedirs(self.outdir, exist_ok=True)
        filename = f"epoch_{epoch+1}_sample_{idx}.png"
        save_path = os.path.join(self.outdir, filename)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        



class DiceLossBCE(tf.keras.losses.Loss):
    # Note: ChatGPT suggested the binary cross-entropy but dice loss was decided because of its use in the google colab file
    """
    Combined Dice loss and Binary Crossentropy.

    Useful for segmentation tasks where Dice encourages overlap and BCE
    penalizes pixel-wise classification errors.
    """
    def __init__(self, smooth=1e-6, name="dice_loss"):
        """
        Initialize DiceLossBCE.

        Args:
            smooth (float): Small constant to avoid division by zero.
            name (str): Loss name.
        """
        super(DiceLossBCE, self).__init__(name=name)
        self.smooth = smooth
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        """
        Compute combined loss.

        Returns:
            Tensor: scalar loss value (batch-averaged).
        """
        # Dice loss + Binary Crossentropy
        # Dice loss to compare areas 
        # Binary crossentropy to increase the white pixels in the mask as disabling setting all pixels to black for high reward.
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
        """
        Return serializable configuration for this loss.
        """
        cfg = super(DiceLossBCE, self).get_config()
        cfg.update({"smooth": self.smooth})
        return cfg
    



class DiceLoss(tf.keras.losses.Loss):
    """
    Standard Dice loss (1 - Dice coefficient).
    """
    def __init__(self, smooth=1e-6, name="dice_loss"):
        """
        Initialize DiceLoss.

        Args:
            smooth (float): Small smoothing constant.
            name (str): Loss name.
        """
        super(DiceLoss, self).__init__(name=name)
        self.smooth = smooth
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, y_true, y_pred):
        """
        Compute combined loss.

        Returns:
            Tensor: scalar loss value (batch-averaged).
        """
        # Dice loss + Binary Crossentropy
        # Dice loss to compare areas 
        # Binary crossentropy to increase the white pixels in the mask as disabling setting all pixels to black for high reward.
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        
        inter = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
        denom = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
        dice = (2. * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1. - dice

        # Print both for debugging
        #tf.print(" BCE:", bce, "Dice:", tf.reduce_mean(dice_loss), summarize=5)

        return tf.reduce_mean(dice_loss)
    
    def get_config(self):
        """
        Return serializable configuration for this loss.
        """
        cfg = super(DiceLoss, self).get_config()
        cfg.update({"smooth": self.smooth})
        return cfg



