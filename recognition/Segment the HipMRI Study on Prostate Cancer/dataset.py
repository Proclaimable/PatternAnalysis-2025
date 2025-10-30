"""
dataset.py

Description:
    Utilities for loading, preprocessing, and preparing TensorFlow image datasets
    for semantic segmentation tasks. This module centralizes common dataset
    operations such as resizing, batching, normalization, mask binarization,
    train/validation splitting, shuffling and prefetching. It is intended to be
    used as a lightweight helper around tf.keras.preprocessing.image_dataset_from_directory
    and tf.data pipeline primitives when building segmentation training loops.

Usage:
    Segmentation Dataset: this class is for accessing data sets. It provides methods processing datasets
    Mainly using the process dataset method where the method will load, preprocess, split the dataset into batches and training/validation sets.
"""
import tensorflow as tf


# ----------------
# Constants
# ----------------
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 10
COLOR_MODE = "rgb"


class SegmentationDataset():
    """
    Helper wrapper around tf.data / image_dataset_from_directory for segmentation.

    Attributes:
        dataset: optional reference to a tf.data.Dataset (not required at init).
        split (float): fraction (0..1) used to split dataset into train/val in process_* methods.
    """
    def __init__(self, dataset, split):
        """
        Initialize the SegmentationDataset helper.

        Args:
            dataset (tf.data.Dataset or None): dataset reference (can be set later).
            split (float): fraction between 0 and 1 used to split dataset into train/validation.
        """
        self.dataset = dataset
        self.split = split

    def process_masks(self, masks):
        """
        Normalize and binarize mask images.

        Converts mask pixel values to float in [0,1] and thresholds at 0.5 so
        outputs are strict binary masks (0.0 or 1.0).

        Args:
            masks (Tensor): uint8 or numeric mask tensor (H,W,C) or batched (B,H,W,C).

        Returns:
            Tensor: float32 tensor with values {0.0, 1.0}.
        """
        # casts masks to floats and normalizes them to values [0, 1]
        # sets values either to 1 or 0
        masks = tf.cast(masks, tf.float32) / 255.0
        masks = tf.where(masks > 0.5, 1.0, 0.0)
        return masks


    def process_dataset(self, datadir, color_mode=COLOR_MODE):
        """
        Load images from a directory and prepare a normalized batched dataset.

        Uses tf.keras.preprocessing.image_dataset_from_directory to read images
        (no labels). Normalizes pixel values to [0,1], applies prefetching, and
        splits into train/validation by batch-count using self.split.

        Args:
            datadir (str): directory containing images (subfolders expected by API).
            color_mode (str): "rgb" or "grayscale".

        Returns:
            (tf.data.Dataset, tf.data.Dataset): (train_dataset, val_dataset)
        """
        # Load batched dataset from directory (no labels, returns batches of images)
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            datadir,
            labels=None,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=color_mode,
            shuffle=False,
        )
        #normalize using image size to values [0, 1]
        ds = ds.map(lambda x: x / 255.0, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        # splits the batch into a train and validation set
        train_dataset = ds.take(int(len(ds) * self.split))
        val_dataset = ds.skip(int(len(ds) * self.split))
        return train_dataset, val_dataset
    
    def process_dataset_masks(self, datadir, color_mode=COLOR_MODE):
        """
        Load mask images from directory and produce binarized batched dataset.

        Similar to process_dataset but applies process_masks to binarize masks.

        Args:
            datadir (str): directory containing mask images.
            color_mode (str): "rgb" or "grayscale" (masks usually grayscale).

        Returns:
            (tf.data.Dataset, tf.data.Dataset): (train_masks, val_masks)
        """
        # Load the image and process with image size batch size and rgb
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            datadir,
            labels=None,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode=color_mode,
            shuffle=False,
        )
        #normalize using image size to values [0, 1] and binarizes the masks
        ds = ds.map(lambda x:  self.process_masks(x), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        train_dataset = ds.take(int(len(ds) * self.split))
        val_dataset = ds.skip(int(len(ds) * self.split))
        return train_dataset, val_dataset

    def map_dataset(self, dataset):
        """
        Apply a mapping function to dataset elements.

        Expects dataset elements shaped (images, masks). This method currently
        forwards to self.process_pair (assumed to be implemented elsewhere).

        Args:
            dataset (tf.data.Dataset): dataset yielding (images, masks)

        Returns:
            tf.data.Dataset: mapped dataset (num_parallel_calls set to AUTOTUNE).
        """
        # Expect a dataset of (images, masks); apply process_pair to each element
        return dataset.map(lambda images, masks: self.process_pair(images, masks), num_parallel_calls=tf.data.AUTOTUNE)

    def shuffle_dataset(self, dataset):
        """
        Shuffle a dataset with a reasonable buffer size.

        Args:
            dataset (tf.data.Dataset): dataset to shuffle.

        Returns:
            tf.data.Dataset: shuffled dataset.
        """
        # lightweight shuffle wrapper
        return dataset.shuffle(buffer_size=1000)

    def prefetch_dataset(self, dataset):
        """
        Prefetch dataset to improve input pipeline throughput.

        Args:
            dataset (tf.data.Dataset): dataset to prefetch.

        Returns:
            tf.data.Dataset: prefetched dataset.
        """
        # prefetch to improve pipeline throughput
        return dataset.prefetch(tf.data.AUTOTUNE)