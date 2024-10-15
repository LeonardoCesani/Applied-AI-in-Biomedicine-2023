
# Classification of PACs and PVCs in PPG using Convolutional Neural Networks

## Project Title
Classification of Premature Atrial Contractions (PACs) and Premature Ventricular Contractions (PVCs) in Photoplethysmography (PPG) signals using Convolutional Neural Networks (CNNs)

## Authors
- **Leonardo Cesani**: leonardo.cesani@mail.polimi.it
- **Martina Fervari**: martina.fervari@mail.polimi.it
- **Andrea Zanette**: andrea.zanette@mail.polimi.it


## Project Overview
This project aims to develop a classification model to differentiate between normal heartbeats and abnormal heartbeats, specifically focusing on Premature Atrial Contractions (PACs) and Premature Ventricular Contractions (PVCs) using PPG signals. These arrhythmias occur when abnormal electrical impulses disturb the heart's rhythm, with PACs originating in the atria and PVCs in the ventricles.

### Goals
1. To develop a machine learning pipeline using convolutional neural networks (CNN) for automated classification of PACs and PVCs from PPG data.
2. To improve the model's classification accuracy through preprocessing, feature extraction, and data balancing.

### Dataset
The dataset used consists of 105 PPG signals obtained from individual patients. These signals were sampled at different frequencies:
- **62 signals at 128 Hz**
- **43 signals at 250 Hz**

For each patient, the dataset contains:
- **PPG signal files**: Raw PPG data with the sampling frequency.
- **Annotations**: Beat annotations labeled as 'N' (Normal), 'V' (Ventricular), and 'S' (Supraventricular).
- **Systolic Peak Position**: Markers for each systolic peak in the PPG signal.

The dataset exhibits significant skewness towards the 'N' (Normal) class, which represents the majority of beats, while abnormal beats ('V' and 'S') are underrepresented, especially in signals sampled at 250 Hz.

## Pre-Processing
To ensure the data was suitable for the model, several pre-processing steps were carried out:
1. **Noise Removal Using Denoising Autoencoders**: A denoising autoencoder was employed to filter out noisy data and detect artifacts. A subsequent reconstruction autoencoder was used to detect outliers that weren't eliminated in the earlier step.
2. **Beat Segmentation**: Each PPG signal was segmented based on systolic peaks into individual beats.
3. **Feature Extraction**: Morphological features such as kurtosis, skewness, and amplitude were calculated to distinguish between different beat types. These features, along with the raw signals, were fed into the model.
4. **Data Normalization**: All beats were normalized to a consistent scale, and noise artifacts were filtered using a Butterworth band-pass filter.

## Model Architecture
The classification model utilizes a **ResNet-based Convolutional Neural Network (CNN)**, which is effective in handling deep networks through the use of residual connections. Additionally, the model includes:
- **Attention Mechanism**: The CBAM (Convolutional Block Attention Module) enhances model performance by focusing on important regions of the PPG signal and specific channels.
- **Features Integration**: Features extracted from the signal are concatenated with the raw signal data to improve classification accuracy.

### Training and Optimization
The model's hyperparameters were selected using hyperparameter tuning on a validation dataset. The loss function used was **Kullback-Leibler Divergence**, which measures the divergence between predicted and actual probability distributions.

### Evaluation
To ensure generalization, a **10-fold cross-validation** approach was used. The model showed good performance in distinguishing between normal and abnormal beats, but further challenges arose in differentiating PACs from PVCs. 

## Key Files
- `assignment.pdf`: Contains the description of the project.
- `Report.pdf`: Contains the report of the project.
- `PPG.ipynb`: Contains the script for data loading and first pre-processing.
- `Denoising Autoencoder.ipynb`: Contains the architecture to train and test the denoising autoencoder used to detect outliers.
- `Preprocessing.ipynb`: Contains functions for the extraction of features and the pre-processing of PPG signals.
- `Train.ipynb`: Contains the model architecture and training process.

## Results
- **Normal vs Abnormal**: The model demonstrated high precision and recall when classifying normal heartbeats from abnormal ones, achieving a precision of ~97% for normal beats.
- **PAC vs PVC**: The classification of PACs and PVCs was more challenging due to the dataset imbalance and morphological similarities between abnormal beats. A **bagging approach** was introduced to stabilize the model, but performance remains limited, particularly for underrepresented classes.

## References
1. **Vincent et al. (2008)**: Denoising autoencoders were used as the base for noise removal in PPG signals. [Extracting and composing robust features with denoising autoencoders](https://doi.org/10.1145/1390156.1390294).
2. **He et al. (2015)**: The ResNet architecture was adapted for one-dimensional signals in this project. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
3. **Woo et al. (2018)**: The CBAM attention block was incorporated to improve the performance of the CNN model. [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521).
4. **Dhar et al. (2022)**: Morphological features such as kurtosis and skewness were employed to capture PPG signal characteristics. [Effortless detection of premature ventricular contraction using computerized analysis of PPG signals](https://link.springer.com/article/10.1007/s12046-021-01693-1).

