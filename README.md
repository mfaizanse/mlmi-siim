# SIIM-ACR Pneumothorax Segementation

## Setup

1. Dataset can be downloaded from Kaggle (see references).

2. The source code is in 'jupyter notebook' directory.

3. The 'SIIM-ACR Dataset Reading' jupyter notebook can be used to first read the dataset files and make the data split into train, validation and test sets.

4. We used Google Colab for model training. Upload the dataset files to google drive and set the corresponding paths in the notebook.

## Overview
We implmented multiple models inlcuding U-Net, Attention U-Net, Tiramisu: FCDenseNet-103 and Tiramisu: FCDenseNet-103 with Attention Gates.

## References:

* Kaggle: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview

* Dataset: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data

* Attention U-Net: Learning Where to Look for the Pancreas, CoRR, 2018  klemek için tıklayın

* The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation, (Simon Jégou, Michal Drozdzal, David Vazquez, Adriana Romero, Yoshua Bengio)

### Code is inspired from:

* https://github.com/baldassarreFe/pytorch-densenet-tiramisu

* https://github.com/ozan-oktay/Attention-Gated-Networks

