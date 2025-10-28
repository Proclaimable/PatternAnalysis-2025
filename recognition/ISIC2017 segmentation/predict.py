"""
predict.py

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
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

import train

def denormalize_image_tf(image):
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image) + 1e-8)
    return image


def evaluate_val_dataset(model, paired_val):
    val_count = tf.data.experimental.cardinality(paired_val).numpy()
    if val_count == tf.data.UNKNOWN_CARDINALITY:
        val_count = sum(1 for _ in paired_val)

    images, masks = next(iter(paired_val.unbatch().batch(val_count)))
    preds = model.predict(images, verbose=0)
    preds_binary = tf.cast(preds > 0.5, tf.float32)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds_binary, masks), tf.float32))

    intersection = tf.reduce_sum(preds_binary * masks)
    union = tf.reduce_sum(preds_binary) + tf.reduce_sum(masks)
    dice = (2.0 * intersection) / (union + 1e-7)

    
    return accuracy, dice

def show_predictions(model, paired_val, title="Binary Segmentation Results (Normalized Color)", n=3):
    """Show model predictions vs ground truth for normalized color-based binary segmentation."""

    rows = 3
    cols = n
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    fig.suptitle(title, fontsize=16)

    val_dataset_unzipped = paired_val.map(lambda x, y: x)
    val_mask_dataset_unzipped = paired_val.map(lambda x, y: y)
    i = 0 

    print(f"Validation dataset size: {len(val_dataset_unzipped)}, Mask dataset size: {len(val_mask_dataset_unzipped)}")


    for img_batch, mask_batch in zip(val_dataset_unzipped.take(3), val_mask_dataset_unzipped.take(3)):

        pred = model.predict(img_batch)
        

        img = img_batch[i]
        mask = mask_batch[i]
        pred = pred[i]


        mask = tf.squeeze(mask, axis=-1)
        pred = tf.squeeze(pred, axis=-1)
        
        pred_binary = tf.cast(pred > 0.5, tf.float32)

        img = denormalize_image_tf(img)
        mask = denormalize_image_tf(mask)

        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Mask {i+1}')
        axes[1, i].axis('off')

        axes[2, i].imshow(pred_binary, cmap='gray')
        axes[2, i].set_title(f'Prediction {i+1}')
        axes[2, i].axis('off')

        i += 1
        if i >= n:
            break
    plt.tight_layout()
    plt.show()

    dice, accuracy = evaluate_val_dataset(model, paired_val)

    print(f"Validation (whole batch) — Dice: {dice:.4f}, Accuracy: {accuracy:.4f}")




def plot_training_history(history):
        # Plot training & validation loss values
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.show()



def main():
    
    history, model, paired_val = train.train(visualiseNum=10, showExample=True)
    plot_training_history(history)
    show_predictions(model, paired_val)


if __name__ == "__main__":
    main()
