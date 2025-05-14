# Automated Image Analysis in Biodiversity Research

This repository contains the complete codebase and experimental workflow accompanying the master's thesis **"Automated Image Analysis in Biodiversity Research"**. The thesis investigates the use of deep learning and content-based image retrieval (CBIR) for the automated analysis of wildlife imagery from camera traps, focusing on classification performance, feature-based retrieval, and the impact of image quality and environmental factors on system robustness.

## Dataset

The dataset, trained models, and relevant JSON files are available on Kaggle. 

These data need to be downloaded and integrated into this repository for full functioning:  

[Wildlife Trap Images (Kaggle Dataset)](https://www.kaggle.com/datasets/katzn13/wildlife-trap-images)

Original images were sourced from the LILA Science repository. The script [`image_loader.py`](image_loader.py) is used to reorganize the raw image directories, which contain inconsistently nested subfolders, into a cleaner and more structured format suitable for further processing.

To execute the script, the data must first be downloaded from the [LILA Science website](https://lila.science/datasets/desert-lion-conservation-camera-traps/), specifically the files named:

- `desert_lion_camera_traps.json` (metadata in JSON format)
- `desert-lion-camera-traps-images/` (directory containing the images)


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

## GUI - CBIR for Biodiversity Research in the Desert of Northern Namibia

This project includes a **Streamlit-based graphical user interface (GUI)** that allows interactive testing of the content-based image retrieval (CBIR) system developed as part of this thesis.

### ✨ Features

- **Upload Query Image**: Upload any image from the test set for similarity search.
- **Model Selection**: Choose between the three best-performing models:
  - GoogLeNet (batch size 128 + augmentation)
  - ResNet50 (batch size 16 + augmentation + iNaturalist)
  - ViT (batch size 64 + augmentation + iNaturalist)
- **Top-k Retrieval**: Set the number of similar images (k) to retrieve.
- **Cosine Similarity**: Results are ranked based on cosine similarity between deep features.
- **Classification Suggestion**: The GUI proposes a class label based on the most frequent categories among the k nearest neighbors, including percentage values and uncertainty notices in case of ties.
- **Species List**: An expandable list shows all 37 supported wildlife categories (scientific and English names).
- **Result Visualization**: Retrieved images are displayed with their corresponding class label and similarity score.

### 🚀 How to Run

1. Install the required dependencies in the requirements.txt file with Python 3.10.11.


2. Start the app from the project folder:

    ```bash
    streamlit run app.py
    ```

3. Open the link shown in the terminal to access the GUI in your browser.
