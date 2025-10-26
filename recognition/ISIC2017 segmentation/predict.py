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

def denormalize_image_tf(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
                # img: tf.Tensor or numpy, either HWC or CHW. Returns HWC numpy array in [0,1].
                img_tf = tf.convert_to_tensor(img, dtype=tf.float32)
                # if CHW -> convert to HWC
                if img_tf.shape.rank == 3 and img_tf.shape[0] == 3:
                    img_tf = tf.transpose(img_tf, [1, 2, 0])
                mean_t = tf.constant(mean, dtype=tf.float32)
                std_t = tf.constant(std, dtype=tf.float32)
                denorm = img_tf * std_t + mean_t
                denorm = tf.clip_by_value(denorm, 0.0, 1.0)
                return denorm.numpy()

def show_predictions(model, dataset, title="🎯 Binary Segmentation Results (Normalized Color)", n=3):
    """Show model predictions vs ground truth for normalized color-based binary segmentation."""
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    with tf.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]

            # Predict with sigmoid model
            pred = model(image.unsqueeze(0))
            # Get pet class probability and convert to binary
            pred_pet_prob = pred[0, 0].numpy()  # Pet class probability
            pred_binary = (pred_pet_prob > 0.5).astype(int)  # Binary prediction

            # Denormalize image for visualization
            # TensorFlow denormalize helper (place this above your function or in module scope)
            

            # Replacement for the $SELECTION_PLACEHOLDER$
            img_show = denormalize_image_tf(image)

            # Show original color image (transpose from CHW to HWC for matplotlib)
            img_display = img_show.permute(1, 2, 0).numpy()  # CHW -> HWC
            axes[0, i].imshow(img_display)
            axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
            axes[0, i].axis('off')

            # Show ground truth binary mask
            axes[1, i].imshow(true_mask, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[1, i].set_title(f'Ground Truth {i+1}', fontweight='bold')
            axes[1, i].axis('off')

            # Show prediction
            axes[2, i].imshow(pred_binary, cmap='RdYlBu_r', vmin=0, vmax=1)
            accuracy = np.mean(pred_binary == true_mask.numpy())
            axes[2, i].set_title(f'Prediction {i+1} (Acc: {accuracy:.2f})', fontweight='bold')
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

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
    history, model, val_dataset = train.train()
    plot_training_history(history)
    show_predictions(model, val_dataset)


    
