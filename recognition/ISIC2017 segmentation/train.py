"""
train.py

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
import matplotlib.pyplot as plt


import modules
import dataset




# ---------------
# checks 
# ---------------
print(tf.__version__)      # TensorFlow version
print(tf.test.is_built_with_cuda())  # CUDA support
print(tf.config.list_physical_devices('GPU'))  # Available GPUs
# ----------
# Constants
# ---------
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 5
COLOR_MODE = "grayscale"
LEAKY_RELU_ALPHA = 0.2
EPOCHS = 100
DROPOUT_P = 0.2
LEARNING_RATE = 1e-4
SEED = 42

tf.random.set_seed(SEED)

# ------------
# Loading Datasets
# ------------
def train(visualiseNum = None):
    ISICdataset = dataset.SegmentationDataset("Dataset\ISIC-2017_Training_Data\Jpeg", split=0.8)
    train_dataset, val_dataset = ISICdataset.process_dataset("Dataset\ISIC-2017_Training_Data\Jpeg")

    ISICmaskdataset = dataset.SegmentationDataset("Dataset\ISIC-2017_Training_Part1_GroundTruth", split=0.8)
    train_mask_dataset, val_mask_dataset = ISICmaskdataset.process_dataset("Dataset\ISIC-2017_Training_Part1_GroundTruth")

    paired_train = tf.data.Dataset.zip((train_dataset, train_mask_dataset))
    paired_train = ISICdataset.map_dataset(paired_train)
    paired_train = ISICdataset.shuffle_dataset(paired_train)
    paired_train = ISICdataset.prefetch_dataset(paired_train)

    paired_val = tf.data.Dataset.zip((val_dataset, val_mask_dataset))
    paired_val = ISICdataset.map_dataset(paired_val)
    paired_val = ISICdataset.shuffle_dataset(paired_val)
    paired_val = ISICdataset.prefetch_dataset(paired_val)




    model = modules.Unet().get_model()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss = modules.DiceLoss()
    )

    history = model.fit(
            paired_train,
            epochs=EPOCHS,
            validation_data=paired_val,
            verbose=1,
            #callbacks=[modules.ShowPredictions(paired_val, n=3, visualize_every=visualiseNum)]
        )

    return history, model, paired_val

def main():
    train()

if __name__ == "__main__":
    main()




