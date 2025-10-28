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
import random


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

COLOR_MODE = "grayscale"
LEAKY_RELU_ALPHA = 0.2
EPOCHS = 150
DROPOUT_P = 0.2
LEARNING_RATE = 1e-4
SEED = 42
SMOOTH = 1e-6

# ------------
# Loading Datasets
# ------------
def train(visualiseNum = None, showExample = False):
    ISICdataset = dataset.SegmentationDataset("Dataset\ISIC-2017_Training_Data\Jpeg", split=0.8)
    train_dataset, val_dataset = ISICdataset.process_dataset("Dataset\ISIC-2017_Training_Data\Jpeg")

    ISICmaskdataset = dataset.SegmentationDataset("Dataset\ISIC-2017_Training_Part1_GroundTruth", split=0.8)
    train_mask_dataset, val_mask_dataset = ISICmaskdataset.process_dataset("Dataset\ISIC-2017_Training_Part1_GroundTruth", color_mode="grayscale")

    paired_train = tf.data.Dataset.zip((train_dataset, train_mask_dataset))
    paired_train = paired_train.shuffle(buffer_size=1024)
    paired_train = paired_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    paired_val = tf.data.Dataset.zip((val_dataset, val_mask_dataset))
    paired_val = paired_val.prefetch(buffer_size=tf.data.AUTOTUNE)

    if showExample == True:
        
        for img_batch, mask_batch in paired_train.take(3):
            img = img_batch[0].numpy()
            mask = mask_batch[0].numpy().squeeze()

            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1)
            plt.imshow(img.squeeze(), cmap='gray' if img.shape[-1]==1 else None)
            plt.title("Image")
            plt.axis('off')

            plt.subplot(1,2,2)
            plt.imshow(mask, cmap='gray')
            plt.title("Mask")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    model = modules.Unet(base = 32).get_model()

    model.compile(
        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE),
        loss = modules.DiceLossBCE()
    )
    
    history = model.fit(
            paired_train,
            epochs=EPOCHS,
            validation_data=paired_val,
            verbose=1,
            batch_size=dataset.BATCH_SIZE,
            callbacks=[modules.ShowPredictions(paired_val, n=3, visualize_every=visualiseNum)]
        )
    
    '''
    history = model.fit(
    paired_train.take(1).repeat(),
    steps_per_epoch=5,
    epochs=20,
    verbose=1,
    callbacks=[modules.ShowPredictions(paired_val, n=3, visualize_every=visualiseNum)]
    )
    
        imgs, masks = next(iter(paired_train.take(1)))
        print(imgs.shape, masks.shape)

        preds = model.predict(imgs)
        print("Prediction shape:", preds.shape)
            
            # Pick the first sample
        img = imgs[0].numpy()
        mask = masks[0].numpy()
        pred = preds[0]

        # Squeeze to remove last channel if needed
        mask = np.squeeze(mask)
        pred = np.squeeze(pred)

        # Threshold prediction to binary mask
        pred_binary = (pred > 0.5).astype(float)

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(img.squeeze(), cmap='gray' if img.shape[-1]==1 else None)
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(pred_binary, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

'''
    return history, model, paired_val

def main():
    history, model, paired_val = train(visualiseNum=1, showExample=False)
    



if __name__ == "__main__":
    main()




