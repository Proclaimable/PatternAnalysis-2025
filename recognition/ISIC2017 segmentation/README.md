# COMP3710 Project: type of AI 
#### Student: s48824150
#### Name: Phoenix Macorkanis

#### Description:
Segment the HipMRI Study on Prostate Cancer (see Appendix for link) using the processed 2D slices (2D
images) available on Rangpur with the 2D UNet [3] with all labels having a minimum Dice similarity
coefficient of 0.75 on the test set on the prostate label. You will need to load Nifti file format and sample
code is provided in Appendix B. [Easy Difficulty]


## 1. Introduction
This report is for the final submission of comp3710 corse and the aim to to experiment with training a improved Unet model to segment medical images namily prostate cancer. The model uses the tensorflow kera modules as a backbone for the code. The report is based of the learnings of the content within Comp3710. The improved Unet model had an accuracy of 

## 2. Dependencies 

- Python 3.10.18
- tensorflow 2.18
- numpy 2.0.2
- tensorboard 2.18
- keras 3.11.2

## 4. Model Architecture
The model architechture was based of the Unet design provied by an article "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BraTS 2017 Challenge" [1]. 

![Improved_Unet](./Photos/Improved_Unet_model.png)

The project idea was taken from "U-Net: Convolutional Networks for Biomedical Image Segmentation"[2] article. This article completes a very similar challenge for medical imaging segmentationof Drosophila first instar larva ventral nerve cord. 

![Unet_model](./Photos/Unet_model.png)

Code for the prediction, loss function and data loading were inspired by the pytorch code for Unet segmentation avaliable on google colab. [3] “3D Improved UNet for Prostate Segmentation



## 5. Data Set
The data set was provided by the ISIC 2017 challenge where they have multiple samples of skin melanoma and ground truths to match. This was used as a paired zip within the network so the loss funciton can evaluate the predictions. The data set is processed within dataset.py for where it 
- normalises; averages the pixels the images by dividing by 255 to make it a [0,1] value
- splits; creates a split for validation and training to test the model on unseen data
- shuffles; improves generatation of the model as it randomises the order of each image in the dataset for each epoch
- prefetches; loads the data early to reduce load time in later interations. 


## 6. Hyperparameters 

## 7. Training

evidence in graph formate of trainning

## 8. Validation 
how did you validate the model

## 9. Results
graphs of the output

## 10. Decussion


## 11. References 
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BraTS 2017 Challenge,” arXiv preprint arXiv:1802.10508v1, Feb. 2018.

[2] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” arXiv preprint arXiv:1505.04597v1, May 2015.

[3] X. Zhou, “3D Improved UNet for Prostate Segmentation,” Google Colab Notebook, Oct. 2025. Available: https://colab.research.google.com/drive/1VOsZSyRhyuHLmgoqGriQk01ub4bKNmZ1?usp=sharing

[4] Codella N, Gutman D, Celebi ME, Helba B, Marchetti MA, Dusza S, Kalloo A, Liopyris K, Mishra N, Kittler H, Halpern A. "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)". arXiv: 1710.05006 [cs.CV]

[5] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics — Survival Prediction: Contribution to the BRATS 2017 Challenge,” arXiv preprint arXiv:1802.10508v1, Feb. 2018.

