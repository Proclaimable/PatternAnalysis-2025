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
import matplotlib.pyplot as plt
import numpy as np

import train

def denormalize_image_tf(image):
    """
    Normalize a tensor to the [0, 1] range for visualization.

    Args:
        image (tf.Tensor): Input image tensor (any numeric range).

    Returns:
        tf.Tensor: Image tensor scaled to [0, 1]. Numerical stability is
                    ensured by a small epsilon in the denominator.
    """
    # Denormalize image tensor from [0,1] to [0,255] for plotting using rgb colormap
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image) + 1e-8)
    return image


def evaluate_val_dataset(model, paired_val):
    """
    Evaluate model on a validation dataset (full-scan) and compute metrics.

    This collects the entire validation dataset (unbatched then rebatched to
    a single batch), runs model.predict once, then computes pixelwise
    accuracy and Dice coefficient.

    Args:
        model (tf.keras.Model): Trained model used for inference.
        paired_val (tf.data.Dataset): Dataset yielding (image_batch, mask_batch).

    Returns:
        tuple: (accuracy, dice) where both are tf.Tensor scalars (float32).
    """
    # Create a final evaluation of the validation dataset using diceloss and pixelwise accuracy
    val_count = tf.data.experimental.cardinality(paired_val).numpy()
    if val_count == tf.data.UNKNOWN_CARDINALITY:
        val_count = sum(1 for _ in paired_val)

    # unbatches the dataset
    images, masks = next(iter(paired_val.unbatch().batch(val_count)))
    # preds using the model
    preds = model.predict(images, verbose=0)
    # cast preds to binary
    preds_binary = tf.cast(preds > 0.5, tf.float32)

    # accuracy calculation
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds_binary, masks), tf.float32))

    # diceloss calculation
    intersection = tf.reduce_sum(preds_binary * masks)
    union = tf.reduce_sum(preds_binary) + tf.reduce_sum(masks)
    dice = (2.0 * intersection) / (union + 1e-7)

    
    return accuracy, dice

def show_predictions(model, paired_val, title="Binary Segmentation Results (Normalized Color)", n=3):
    """
    Display a small grid of original images, ground-truth masks and predictions.

    The function expects `paired_val` to yield batches (image_batch, mask_batch).
    It takes up to `n` examples (from up to 3 batches in the current implementation)
    and shows original, mask, and binary prediction in a 3 x n grid.

    Args:
        model (tf.keras.Model): Model used to generate predictions.
        paired_val (tf.data.Dataset): Dataset yielding (image_batch, mask_batch).
        title (str): Figure title.
        n (int): Number of columns / examples to show (max 3 in current loop).

    Notes:
        - This function performs tf.squeeze on masks/predictions only when their
        last dimension equals 1 (grayscale). RGB channels are preserved.
        - Converting tensors to numpy is done for matplotlib display.
    """

    # creates subplots of final results after training the model
    rows = 3
    cols = n
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    fig.suptitle(title, fontsize=16)

    val_dataset_unzipped = paired_val.map(lambda x, y: x)
    val_mask_dataset_unzipped = paired_val.map(lambda x, y: y)
    i = 0 

    print(f"Validation dataset size: {len(val_dataset_unzipped)}, Mask dataset size: {len(val_mask_dataset_unzipped)}")


    for img_batch, mask_batch in zip(val_dataset_unzipped.take(3), val_mask_dataset_unzipped.take(3)):
        # ploting stuff
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
    """
    Plot training and validation loss curves from a Keras History object.

    Args:
        history (tf.keras.callbacks.History): History returned by model.fit.
    """
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

def plot_val_metrics(model, val_dataset, threshold=0.5):
    """
    Compute per-sample accuracy and Dice on a validation dataset and plot results.

    Args:
        model (tf.keras.Model): Model for inference.
        val_dataset (tf.data.Dataset): Dataset yielding (x_batch, y_batch).
        threshold (float): Threshold to binarize model probabilities.

    Returns:
        tuple: (mean_acc, mean_dice) as numpy floats.
    """
    # creates a plot for all data in the validation dataset as a scatter plot
    y_true, y_pred = [], []
    for x_batch, y_batch in val_dataset:
        preds = model.predict(x_batch, verbose=0)
        y_true.append(y_batch.numpy())
        y_pred.append(preds)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_pred_bin = (y_pred > threshold).astype(np.float32)

    acc = np.mean(y_true == y_pred_bin, axis=(1,2,3))
    dice = (2 * np.sum(y_true * y_pred_bin, axis=(1,2,3))) / (
        np.sum(y_true, axis=(1,2,3)) + np.sum(y_pred_bin, axis=(1,2,3)) + 1e-7
    )

    mean_acc = np.mean(acc)
    mean_dice = np.mean(dice)

    plt.figure(figsize=(10,4))
    plt.scatter(range(len(acc)), acc, alpha=0.7)
    plt.axhline(mean_acc, color='r', linestyle='--')
    plt.title(f'Pixelwise Accuracy per Image (Mean={mean_acc:.4f})')
    plt.xlabel('Sample Index')
    plt.ylabel('Pixelwise Accuracy')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,4))
    plt.scatter(range(len(dice)), dice, alpha=0.7)
    plt.axhline(mean_dice, color='r', linestyle='--')
    plt.title(f'Dice Accuracy per Image (Mean={mean_dice:.4f})')
    plt.xlabel('Sample Index')
    plt.ylabel('Dice Accuracy')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

    return mean_acc, mean_dice




def main():
    """
    Example entry point: train a model (via train.train), then plot history and
    show validations/predictions. Used when running this module as script.

    Note:
        train.train must return (history, model, paired_val) as expected.
    """
    
    history, model, paired_val = train.train(visualiseNum=5, showExample=False)
    plot_training_history(history)
    show_predictions(model, paired_val)
    plot_val_metrics(model, paired_val)
    



if __name__ == "__main__":
    main()
