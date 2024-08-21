# Lung-Segmentation
Lung Segmentation using U-Net with PyTorch Lightning
This repository contains a deep learning pipeline for lung segmentation using the U-Net architecture, implemented with PyTorch and PyTorch Lightning. The model was trained and tested on the https://challenge.isic-archive.com/data/#2017, achieving a Dice score of 0.79.

Features:
U-Net Architecture: A custom U-Net model with a ResNet-50 encoder backbone, capable of segmenting lung regions from chest X-ray images.
Data Augmentation: On-the-fly data augmentation is applied during training to improve model robustness. Augmentations include random horizontal/vertical flips, rotations, brightness, and contrast adjustments.
Loss Function: A combination of Binary Cross-Entropy (BCE) and Dice loss is used to optimize the model, balancing pixel-wise accuracy with overlap-based performance.
Model Checkpointing: The model is trained using PyTorch Lightning, with automatic checkpointing based on validation Dice score.
Visualization: The repository includes functions for visualizing sample images, ground truth masks, and model predictions.
Results:
Dice Score: The model achieved a Dice score of 0.79 on the test set from the ISIC 2017 Challenge dataset.
Usage:
Data Preparation: Organize your dataset into images and masks directories.
Training: Adjust parameters such as learning rate, batch size, and the number of epochs in the main function and start training.
Testing: Once training is complete, the model can be tested using the provided test dataset.
Pretrained Weights: The model supports loading pretrained weights for fine-tuning.
Requirements:
PyTorch
PyTorch Lightning
Torchvision
Scikit-learn
Matplotlib
TensorBoard.

This project showcases a robust approach to lung segmentation using deep learning, aiming to assist in medical image analysis and other related tasks.
