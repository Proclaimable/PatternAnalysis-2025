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
import tensorflow as tf
import keras 

from pathlib import Path

# ----------------
# Constants
# ----------------
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 3
COLOR_MODE = "grayscale"

# ----------------
# Functions
# ----------------


def preprocess_image(datadir, shuffle=False):
    
    # Load the image and process with image size batch size and grayscale
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        datadir,
        labels=None,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode=COLOR_MODE,
        shuffle=shuffle
    )
    #normalize using image size 
    ds = ds.map(lambda x: x / 255.0, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def datasplit(dataset: tf.data.Dataset, train_ratio=0.8, test_ratio=0.2):
    train_size = int(len(dataset) * train_ratio)
    test_size = int(len(dataset) * test_ratio)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    return train_dataset, test_dataset

def __getitem__(self, idx):
        # Get image and mask
        image, mask = self.dataset[idx]
        # Apply transforms to image
        if self.transform:
            image = self.transform(image)

        mask = transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask_np = np.array(mask)  # Convert PIL to numpy array - this preserves [1,2,3]
        binary_mask = np.zeros_like(mask_np, dtype=np.uint8)
        binary_mask[mask_np == 1] = 1  # segmentation pixels = 1
        binary_mask[mask_np == 2] = 0  # background pixels = 0
        binary_mask[mask_np == 3] = 0  # border pixels -> background (no ignored pixels)

        # Convert to tensor
        binary_mask = tf.convert_to_tensor(binary_mask, dtype=tf.int64)

        return image, binary_mask
