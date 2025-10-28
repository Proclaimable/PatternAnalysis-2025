"""
dataset.py

Description:
    This module contains various utility functions and helper classes
    used for the [Project Name/Description]. It includes functions for
    [list of core functionalities, e.g., data preprocessing, model training,
    etc.]. The functions are modular and can be easily imported into other
    parts of the project.

    2. ISIC 2017/8 dermoscopic imaging challenge data for skin cancer - This is part of the ISIC 2017/8 challenge
    and comes with segmentation labels and lesion class labels.
    Rangpur Path: /home/groups/comp3710/ISIC2018

Usage:
    - Function 1: [Brief description of how to use]
    - Function 2: [Brief description of how to use]
    - Etc.

Notes:
    - [Any special notes about the module, e.g., performance considerations]
"""

import numpy as np
from matplotlib import transforms
from sklearn.utils import shuffle
import tensorflow as tf
import keras 

from pathlib import Path

# ----------------
# Constants
# ----------------
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
COLOR_MODE = "rgb"

# ----------------
# Functions
# ----------------




class SegmentationDataset():
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split


    def process_dataset(self, datadir, color_mode=COLOR_MODE):
        # Load the image and process with image size batch size and rgb
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            datadir,
            labels=None,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=color_mode,
            shuffle=False,
        )
        #normalize using image size 
        ds = ds.map(lambda x: x / 255.0, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        train_dataset = ds.take(int(len(ds) * self.split))
        val_dataset = ds.skip(int(len(ds) * self.split))
        return train_dataset, val_dataset

    def process_pair(self, images, masks):
        masks = tf.cast(masks, tf.float32)
        masks = tf.where(tf.equal(masks, 1), tf.ones_like(masks), tf.zeros_like(masks))
        masks = tf.cast(masks, tf.float32)
        return images, masks


    def map_dataset(self, dataset):
        return dataset.map(lambda images, masks: self.process_pair(images, masks), num_parallel_calls=tf.data.AUTOTUNE)

    def shuffle_dataset(self, dataset):
        return dataset.shuffle(buffer_size=1000)

    def prefetch_dataset(self, dataset):
        return dataset.prefetch(tf.data.AUTOTUNE)