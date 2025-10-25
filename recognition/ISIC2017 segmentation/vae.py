"""
Variational Autoencoder (VAE) for Medical Image Segmentation and Latent Space Visualization

This script implements a VAE using TensorFlow/Keras for grayscale medical images.
It includes:
- Data loading and preprocessing (with one-hot encoded masks)
- Encoder and decoder model definitions
- Custom VAE class with train/test steps and Dice coefficient metric
- Training and evaluation routines
- Visualization of reconstructions and latent space (UMAP)

Author: Phoenix
COMP3710 Lab 2
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers
import umap
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import ploting as p

# -------------------------------
# Constants and Hyperparameters
# -------------------------------
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
LATENT_DIM = 16
EPOCHS = 20
NUM_CLASSES = 4

# -------------------------------
# TensorFlow Environment Check
# -------------------------------
print(tf.__version__)      # TensorFlow version
print(tf.test.is_built_with_cuda())  # CUDA support
print(tf.config.list_physical_devices('GPU'))  # Available GPUs

# -------------------------------
# Data Preprocessing Functions
# -------------------------------

def preprocess_mask(mask):
    """
    Converts mask to one-hot encoding for multi-class segmentation.
    Args:
        mask: Tensor of shape (batch, height, width, 1)
    Returns:
        One-hot encoded mask of shape (batch, height, width, NUM_CLASSES)
    """
    mask = tf.cast(mask, tf.int32)
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.one_hot(mask, depth=NUM_CLASSES, dtype=tf.float32)
    return mask

def make_mask_dataset(mask_dir, shuffle=True):
    """
    Loads images from a directory and applies preprocessing.
    Args:
        mask_dir: Directory containing mask images.
        shuffle: Whether to shuffle the dataset.
    Returns:
        Prefetched tf.data.Dataset of (image, mask) pairs.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        mask_dir,
        labels=None,
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )
    ds = ds.map(lambda x: (x/255.0, preprocess_mask(x)), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

# -------------------------------
# Dataset Preparation
# -------------------------------
train_ds = make_mask_dataset("keras_png_slices_data/keras_png_slices_train")
val_ds = make_mask_dataset("keras_png_slices_data/keras_png_slices_validate", shuffle=False)
test_ds = make_mask_dataset("keras_png_slices_data/keras_png_slices_test", shuffle=False)

# -------------------------------
# Model Definitions
# -------------------------------

# Encoder Model
encoder_inputs = keras.Input(shape=(256, 256, 1))
x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)

def sampling(args):
    """
    Reparameterization trick for VAE.
    Args:
        args: (z_mean, z_log_var)
    Returns:
        Sampled latent vector z.
    """
    z_m, z_lv = args
    eps = tf.random.normal(shape=tf.shape(z_m))
    return z_m + tf.exp(0.5 * z_lv) * eps

z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# Decoder Model
latent_inputs = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(16 * 16 * 256, activation="relu")(latent_inputs)
x = layers.Reshape((16, 16, 256))(x)
x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, strides=2, padding="same", activation="tanh")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# -------------------------------
# Dice Coefficient Metric
# -------------------------------

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice coefficient for multi-class segmentation.
    Args:
        y_true: Ground truth one-hot mask.
        y_pred: Predicted mask (logits or probabilities).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Dice coefficient (scalar tensor).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=NUM_CLASSES, dtype=tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

# -------------------------------
# Custom VAE Model
# -------------------------------

class VAE(keras.Model):
    """
    Variational Autoencoder with custom training and testing steps.
    Tracks total loss, reconstruction loss, KL divergence, and Dice coefficient.
    """
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        # List of metrics to be reset at the start of each epoch
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        """
        Custom training step for VAE.
        Args:
            data: Tuple (images, masks) or images only.
        Returns:
            Dictionary of tracked losses.
        """
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1,2)))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def test_step(self, data):
        """
        Custom test/validation step for VAE.
        Args:
            data: Tuple (images, masks) or images only.
        Returns:
            Dictionary of tracked losses and Dice coefficient.
        """
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1,2)))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        dsc = dice_coefficient(data, reconstruction)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "dice_coefficient": dsc
        }

# -------------------------------
# Model Training
# -------------------------------

vae = VAE(encoder, decoder)

vae.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=[dice_coefficient]
)

history = vae.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# -------------------------------
# Model Evaluation
# -------------------------------

results = vae.evaluate(test_ds)
print("All test results:", dict(zip(vae.metrics_names, results)))
print("Test DSC (accuracy):", results[vae.metrics_names.index("dice_coefficient")])

# -------------------------------
# Plotting Loss Curves
# -------------------------------

plt.plot(history.history["reconstruction_loss"], label="train")
if "val_reconstruction_loss" in history.history:
    plt.plot(history.history["val_reconstruction_loss"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Loss")
plt.title("Reconstruction Loss over Epochs")
plt.legend()
plt.show()

# -------------------------------
# Display Reconstructions
# -------------------------------

x, _ = next(iter(test_ds))
z_mu, z_lv, z = encoder.predict(x, verbose=0)
recon = decoder.predict(z, verbose=0)

n = 8
plt.figure(figsize=(n*1.5,3))
for i in range(n):
    plt.subplot(2, n, i+1)
    plt.imshow(tf.squeeze(x[i]), cmap="gray")
    plt.axis("off")
    plt.subplot(2, n, n+i+1)
    plt.imshow(tf.squeeze(recon[i]), cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.show()

# -------------------------------
# UMAP Visualization of Latent Space
# -------------------------------

cap = 5000

# Collect latent vectors for train set
Z_tr, n = [], 0
for batch in train_ds:
    x, _ = batch
    z_mu, z_lv, z = vae.encoder.predict(x, verbose=0)
    Z_tr.append(z_mu)
    n += z_mu.shape[0]
    if n >= cap:
        break
Z_tr = np.concatenate(Z_tr, axis=0)[:cap]

# Collect latent vectors for test set
Z_te, n = [], 0
for batch in test_ds:
    x, _ = batch
    z_mu, z_lv, z = vae.encoder.predict(x, verbose=0)
    Z_te.append(z_mu)
    n += z_mu.shape[0]
    if n >= cap:
        break
Z_te = np.concatenate(Z_te, axis=0)[:cap]

# UMAP dimensionality reduction
reducer = umap.UMAP(n_components=2, random_state=42)
U_tr = reducer.fit_transform(Z_tr)
U_te = reducer.transform(Z_te)

plt.figure(figsize=(7,6))
plt.scatter(U_tr[:,0], U_tr[:,1], s=4, label="train")
plt.scatter(U_te[:,0], U_te[:,1], s=4, alpha=0.6, label="test")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP of z_mean: train vs test")
plt.legend()
plt.tight_layout()
plt.show()

