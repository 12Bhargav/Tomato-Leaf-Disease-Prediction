# Tomato Leaf Disease Prediction

This repository contains a deep learning project for the classification of diseases in tomato plant leaves. The aim is to accurately identify various diseases (or healthy leaves) from images, assisting in early detection and management for agricultural applications.


## Overview

The project develops and evaluates a **Convolutional Neural Network (CNN)**-based model, specifically a hybrid architecture leveraging a **Swin Transformer backbone**, for image classification. Key aspects include:

1.  **Dataset Preparation**: Loading and splitting the image dataset (expected from `/content/drive/MyDrive/PlantVillage`) into training, validation, and test sets.
2.  **Enhanced Data Augmentation**: Applying a comprehensive set of transformations (resizing, random cropping, flips, rotations, color jitter, Gaussian blur, random erasing) to increase data diversity and improve model generalization.
3.  **Hybrid Model Architecture**: Utilizing a pre-trained `swin_tiny_patch4_window7_224` as a feature extractor, combined with a custom convolutional head and a classifier, for robust feature learning and classification.
4.  **Advanced Training Techniques**:
    * **Mixed Precision Training**: Employing `torch.cuda.amp.autocast` and `GradScaler` for faster training and reduced memory consumption.
    * **Learning Rate Scheduling**: Using `OneCycleLR` with AdamW optimizer for efficient convergence.
    * **MixUp Augmentation**: Applying MixUp during early training epochs to further enhance generalization.
    * **Class Weighting and Label Smoothing**: Implementing `CrossEntropyLoss` with class weights and label smoothing for handling class imbalance and improving model robustness.
5.  **Model Evaluation**: Tracking training and validation loss/accuracy, and saving the best performing model.

## Dataset

This project is designed to work with an image dataset typically structured with subfolders representing different classes of tomato leaf health (e.g., healthy, various diseases). The `DATA_DIR` is configured to `/content/drive/MyDrive/PlantVillage`. The `NUM_CLASSES` is set to 10, indicating the number of distinct categories the model is trained to identify.

## Technologies Used

* **Python**: The core programming language.
* **PyTorch**: The deep learning framework used for model building and training.
* **torchvision**: For dataset loading and image transformations.
* **timm (PyTorch Image Models)**: For easily accessing and using pre-trained computer vision models like Swin Transformer.
* **NumPy**: For numerical operations.
* **Matplotlib**: For plotting training metrics.
* **Seaborn**: For enhanced data visualization (e.g., confusion matrix).
* **Scikit-learn**: For classification report and confusion matrix generation.

## Usage

To run this notebook and train your own model:

1.  **Dataset Setup**: Ensure your tomato leaf disease image dataset is organized in the specified `DATA_DIR` (`/content/drive/MyDrive/PlantVillage`), with subfolders for each class.
2.  **Environment Setup**: Install all required Python libraries:
    ```bash
    pip install torch torchvision timm matplotlib seaborn scikit-learn numpy
    ```
    If you plan to use a GPU, ensure you have the appropriate CUDA toolkit installed.
3.  **Open Notebook**: Launch Jupyter Notebook or JupyterLab and open `Tomato_Disease_Leaf_Pred (1).ipynb`.
4.  **Execute Cells**: Run all the cells sequentially. The notebook will handle data loading, preprocessing, model definition, training, and evaluation.
5.  **Monitor Progress**: Observe the training and validation metrics printed per epoch. The best performing model will be saved as `best_model.pth`.
