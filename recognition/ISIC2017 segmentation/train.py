"""
train.py

Description:
    Training entrypoint and utilities for the Improved U-Net segmentation
    project. Loads datasets, constructs the model, configures training
    (optimizer, loss, callbacks) and provides a train(...) function that
    returns (history, model, paired_val) for downstream evaluation and
    visualization.

Usage:
    - train(visualiseNum=None, showExample=False):
        Prepare datasets, compile the ImprovedUnet model, run model.fit and
        return (history, model, paired_val).
    - main():
        Example CLI entrypoint that calls train().

Notes:
    - Expects dataset.SegmentationDataset helpers to provide train/val image
    and mask tf.data.Datasets yielding batches shaped (B, H, W, C).
    - Use tf.keras (TensorFlow's Keras) models/losses for dtype compatibility.
    - Callback ShowPredictions saves example visuals; avoid blocking calls
    (use plt.savefig + plt.close).
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


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
EPOCHS = 500
DROPOUT_P = 0.3
LEARNING_RATE = 1e-4
SMOOTH = 1e-6

# ------------
# Loading Datasets
# ------------
def train(visualiseNum = None, showExample = False):
    """
    Prepare data, build and train the ImprovedUnet model.

    Steps:
    - Load images and masks using SegmentationDataset helpers.
    - Pair, shuffle and prefetch training dataset.
    - Build the model, compile with DiceLossBCE and Adam optimizer.
    - Run model.fit with ShowPredictions callback.

    Args:
        visualiseNum (int or None): If set, passed to ShowPredictions.visualize_every
            to control how often visuals are saved. None disables visualization.
        showExample (bool): If True, display a few example image/mask pairs
            from the training set before training (useful for debugging).

    Returns:
        tuple: (history, model, paired_val)
            history: Keras History object from model.fit.
            model: The trained Keras model.
            paired_val: Validation dataset (tf.data.Dataset) used for validation/visualization.
    """
    # main training funciton


    # loads the data set from preset folders using methods from dataset.py
    ISICdataset = dataset.SegmentationDataset("Nifti files\Scan_images", split=0.8)
    train_dataset, val_dataset = ISICdataset.process_dataset("Nifti files\Scan_images", color_mode="grayscale")

    ISICmaskdataset = dataset.SegmentationDataset("Nifti files\Labels_images", split=0.8)
    train_mask_dataset, val_mask_dataset = ISICmaskdataset.process_dataset_masks("Nifti files\Labels_images", color_mode="grayscale")

    # pairs the datasets into train mask pairs for the model to use as (x,y) comparisons
    paired_train = tf.data.Dataset.zip((train_dataset, train_mask_dataset))
    paired_train = paired_train.shuffle(buffer_size=1024)
    paired_train = paired_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    paired_val = tf.data.Dataset.zip((val_dataset, val_mask_dataset))
    paired_val = paired_val.prefetch(buffer_size=tf.data.AUTOTUNE)

    img_val, mask_val = next(iter(paired_val))
    print(f"shapes img:{img_val.shape}, mask:{mask_val.shape}")

    

    if showExample == True:
        # Visualize some examples from the training set for debugging
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


    # loads the model from modules.py base is the base number of filters
    model = modules.ImprovedUnet(in_channels=1, base = 32).get_model()

    model.compile(
        # Uses Adam optimizer using hyperparameter as the learning rate and premade dicelossBCE function from modules.py
        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE),
        loss = modules.DiceLoss()
    )
    
    # main training loop using fit method from tensorflow keras
    # importantly using the paired_train for training and paired_val for validation

    
    history = model.fit(
            paired_train,
            epochs=EPOCHS,
            validation_data=paired_val,
            verbose=1,
            batch_size=dataset.BATCH_SIZE,
            callbacks=[modules.ShowPredictions(paired_val, n=3, visualize_every=visualiseNum)]
        )
    

    
    return history, model, paired_val

def main():
    history, model, paired_val = train(visualiseNum=1, showExample=True)
    



if __name__ == "__main__":
    main()




