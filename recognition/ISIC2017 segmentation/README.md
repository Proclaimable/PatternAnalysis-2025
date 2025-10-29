# COMP3710 Project: Improved Unet Design
#### Student: s48824150
#### Name: Phoenix Macorkanis

#### Description:
Segment the HipMRI Study on Prostate Cancer (see Appendix for link) using the processed 2D slices (2D
images) available on Rangpur with the 2D UNet [3] with all labels having a minimum Dice similarity
coefficient of 0.75 on the test set on the prostate label.
code is provided in Appendix B. [Easy Difficulty]


## 1. Introduction
This report is for the final submission of COMP3710 course. The aim to to experiment with training a improved U-Net model to segment medical images, specifically prostate cancer. The model uses the tensorflow kera modules as a backbone for the code. The report is based of the learnings of the content within Comp3710. The improved U-Net model had an accuracy of 80% for 100 epochs. Note some problem solving and small suggestions were made with the help of chatGPT - 5 [8] notably with a suggestion to add the BCE loss module in. 

## 2. Dependencies 
The user of the script requires to download cuda tool kit for their GPU through the nivdia site:https://developer.nvidia.com/cuda-downloads. The cuda must be at least version 12. This report used a 3070ti RTX GPU to train the model. 


| Package | Version |
|--------|---------|
| Python | 3.9.23 |
| TensorFlow | 2.10.0 |
| Keras | 2.10.0 |
| NumPy | 1.23.5 |
| TensorBoard | 2.10.1 |
| Matplotlib | 3.9.2 |
| CUDA Toolkit | 11.2.2 |
| cuDNN | 8.1.0 |


The user of the script requires to download cuda tool kit for their GPU through the nivdia site:https://developer.nvidia.com/cuda-downloads. The cuda must be at least version 12. This report used a 3070ti RTX GPU to train the model. 

| Package | Version |
|--------|---------|
| CUDA Toolkit | 11.2.2 |
| cuDNN | 8.1.0 |

note for reproducing the code. Run the predict.py for full useage of all scripts

## 4. Model Architecture
The model architechture was based of the Unet design provied by an article "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BraTS 2017 Challenge" [1]. 

![Improved_Unet](./Photos/Improved_Unet_model.png)

The project idea was taken from "U-Net: Convolutional Networks for Biomedical Image Segmentation"[2] article. This article completes a very similar challenge for medical imaging segmentation of Drosophila first instar larva ventral nerve cord. 

![Unet_model](./Photos/Unet_model.png)

Code for the prediction, loss function and data loading were inspired by the pytorch code for Unet segmentation avaliable on google colab. [3] “3D Improved UNet for Prostate Segmentation

### Loss Funtions
**Dice Similarity Coefficient (DSC):**
$$
DSC = \frac{2|X \cap Y|}{|X| + |Y|}
$$
where X is the prediciton dataset and Y is the ground truths

**Binary Cross-Entropy (BCE):**
$$
BCE = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$
where $yi$ is the ground truths and $pi$ is the prediction.




## 5. Data Set
The data set was provided by the ISIC 2017 challenge where they have multiple samples of skin melanoma and ground truths to match. [4] This was used as a paired zip within the network so the loss funciton can evaluate the predictions. The data set is processed within dataset.py for where it 
- normalises; averages the pixels the images by dividing by 255 to make it a [0,1] value
- splits; creates a split for validation and training to test the model on unseen data
- shuffles; improves generatation of the model as it randomises the order of each image in the dataset for each epoch
- prefetches; loads the data early to reduce load time in later interations. 


## 6. Hyperparameters 

LEAKY_RELU_ALPHA = 0.2  
EPOCHS = 100  
DROPOUT_P = 0.2  
LEARNING_RATE = 1e-4   
SMOOTH = 1e-6  
BATCH_SIZE = 10


## 7. Training

The model was trained using the ground truths to test the prediciton by the model during the epochs. The metric for evaluation was a combination of Dice Loss and Binary Crossentropy assigning them equal weights. Dice loss was the main metric as it compared the coverd areas of prediciton to the ground truth where coe = 1 would be a perfect coverage. This lead to training favouring leaving the whole mask to be black to maximise averages over the dataset. [6] Thus, Binary Crossentropy was added with the same weight to increase the favourability of having white areas in the mask. This evaluates the pixel maping of the prediciton in a logarithmic fashion. [7]

![Training_Graph](./Photos/Training_Graph.png)

Training graph of the loss function over the epochs



## 8. Validation 
The model was vaildated using the validation dataset made in the train.py using dice and pixelwise comparision against the ground truths to get an average dice coeifficent and accuracy number. A plot is made of this comparision against all the validation dataset to spot any outliers from the average.

The model was validated to be "Validation (whole batch) — Dice: 0.9644, Accuracy: 0.8058"

<img src="Photos\Example predicitons.png" width="200" height="300">

example predictions from model



<img src="Photos\Pixelwise_Accuracy.png">

<img src="Photos\Dice_Accuracy.png">

Notably there were some outliers in the validation of the model which could stem from formating of the data when the accuracy is 0 in some cases or addition of extra white space in the masking for the accuracy of the 0.4 - 0.6 cases. However, the general accuracy of most cases were successful as seen in the mean statsistics. If it is not in the fault of processing the dataset. Training the model for longer or training with harder examples could resolve some of the outliers for the accuracy. 




## 9. Discussion
The model performed increasing well taking a long time to produce any overfitting. Further evaluation and refinements of the model including looking the hyperparameters of the learning rates, relu alpha, dropout percent and loss funciton weight. Experiemention of these hyperparameters could create a faster or more accurate model output. Furthermore, training the model till overfitting would also be an interesting task to look into. Lastly, the time to train was heavily increase by the time it took to shuffle the dataset inbetween epochs. This could be due to the model having to read and write from the disc for every shuffle. Some research into a better shuffling method would speed up training the model by significant margins.


## 10. References 
[1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BraTS 2017 Challenge,” arXiv preprint arXiv:1802.10508v1, Feb. 2018.

[2] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” arXiv preprint arXiv:1505.04597v1, May 2015.

[3] X. Zhou, “3D Improved UNet for Prostate Segmentation,” Google Colab Notebook, Oct. 2025. Available: https://colab.research.google.com/drive/1VOsZSyRhyuHLmgoqGriQk01ub4bKNmZ1?usp=sharing

[4] Codella N, Gutman D, Celebi ME, Helba B, Marchetti MA, Dusza S, Kalloo A, Liopyris K, Mishra N, Kittler H, Halpern A. "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)". arXiv: 1710.05006 [cs.CV]

[5] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation and Radiomics — Survival Prediction: Contribution to the BRATS 2017 Challenge,” arXiv preprint arXiv:1802.10508v1, Feb. 2018.

[6] Z. Y. Zheng, B. H. Tian, S. Yu, X. Yang, Q. Yu, J. Zhou, G. Jiang, Q. Zheng, J. Pu and L. Wang, “Adaptive boundary-enhanced Dice loss for image segmentation,” Biomedical Signal Processing and Control, vol. 106, Art. no. 107741, 2025. DOI: 10.1016/j.bspc.2025.107741.

[7] M. Yeung, E. Sala, C.-B. Schönlieb and L. Rundo, “Unified Focal Loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation,” Computerized Medical Imaging and Graphics, vol. 95, Art. no. 102026, 2022. DOI: 10.1016/j.compmedimag.2021.102026.

[8] OpenAI, “ChatGPT (GPT-5),” OpenAI, San Francisco, CA, USA. Available: https://chat.openai.com/.

## Extra photos for presentation
Progress of training

Examples of training at epoch 1,11,21,31

<img src="Photos\epoch_1_sample.png">
<img src="Photos\epoch_11_sample.png">
<img src="Photos\epoch_21_sample.png">
<img src="Photos\epoch_31_sample.png">

