# Automated Image Analysis in Biodiversity Research

This repository contains the complete codebase and experimental workflow accompanying the master's thesis **"Automated Image Analysis in Biodiversity Research"**. The thesis investigates the use of deep learning and content-based image retrieval (CBIR) for the automated analysis of wildlife imagery from camera traps, focusing on classification performance, feature-based retrieval, and the impact of image quality and environmental factors on system robustness.

## Dataset

The dataset, trained models, and relevant JSON files are available on Kaggle:  
[Wildlife Trap Images (Kaggle Dataset)](https://www.kaggle.com/datasets/katzn13/wildlife-trap-images)

Original images were sourced from the LILA Science repository. Images can be downloaded using:

- [`image_loader.py`](image_loader.py)

## Data Preparation and Preprocessing

- [`dataset_balance.ipynb`](dataset_balance.ipynb): Reorganizes the dataset into a balanced directory structure suitable for training and evaluation.
- [`data_scraping.ipynb`](data_scraping.ipynb): Complements underrepresented classes by scraping wildlife images from iNaturalist.
- [`data_augmentation.ipynb`](data_augmentation.ipynb): Applies augmentation techniques to increase dataset diversity and address class imbalance.

## Exploratory Data Analysis

- [`EDA.ipynb`](EDA.ipynb): Overview of class distributions, dataset composition, and basic statistics.
- [`day_night_classification.ipynb`](day_night_classification.ipynb): Classification of day and night images and its impact on model behavior.
- [`challenging_images.ipynb`](challenging_images.ipynb): Identification of low-quality or problematic images using brightness, contrast, and Laplacian variance.
- [`bounding_boxes.ipynb`](bounding_boxes.ipynb): Analysis of bounding box information to assess object size and visibility.

## Model Training and Tuning

- [`JupyterLab_notebooks/training.py`](JupyterLab_notebooks/training.py): Training pipeline using FastAI and pretrained CNN architectures.
- [`Kaggle_notebooks/vit-training.ipynb`](Kaggle_notebooks/vit-training.ipynb): Training routine for the Vision Transformer (ViT).
- Training involved extensive hyperparameter tuning and dataset enhancement using augmented and scraped images.

## Model Evaluation

- [`JupyterLab_notebooks/Evaluation.ipynb`](JupyterLab_notebooks/Evaluation.ipynb)  
- [`JupyterLab_notebooks/GoogLeNet.ipynb`](JupyterLab_notebooks/GoogLeNet.ipynb)  
- [`JupyterLab_notebooks/ResNet50.ipynb`](JupyterLab_notebooks/ResNet50.ipynb)  
- [`JupyterLab_notebooks/ViT.ipynb`](JupyterLab_notebooks/ViT.ipynb)  
- [`JupyterLab_notebooks/classwise_error_analysis.ipynb`](JupyterLab_notebooks/classwise_error_analysis.ipynb)  
- [`JupyterLab_notebooks/Impact of Image Quality and Environment Factors.ipynb`](JupyterLab_notebooks/Impact%20of%20Image%20Quality%20and%20Environment%20Factors.ipynb): Assessment of model robustness with respect to image degradation, occlusion, and lighting conditions.

## Model Explainability

- [`JupyterLab_notebooks/Grad-CAM.ipynb`](JupyterLab_notebooks/Grad-CAM.ipynb): Visualization of class activation maps to interpret model decision-making using Grad-CAM.

## Citation

If you use this repository or dataset in your work, please cite the corresponding thesis and the associated Kaggle dataset.
