# MosMedData COVID-19 CT Scan Classification

This project aims to classify 3D CT Scans of patients into 5 severity levels (CT-0 to CT-4) based on COVID-19 related findings. The model leverages the MosMedData dataset.

## Architecture and Pipeline

The solution uses a hybrid Deep Learning and Machine Learning approach, divided into the following steps:

1. **Data Preprocessing & Loading:**
   - 3D NIfTI CT scans are loaded using `nibabel`.
   - **Hounsfield Unit (HU) Windowing** is applied (`-1000` to `400`) to isolate lung tissue and remove noise like bones and air.
   - To capture 3D spatial information without memory exhaustion, we extract 3 axial slices at 25%, 50%, and 75% depth and stack them into a 3-channel (RGB-like) tensor.
   - The tensor is normalized between 0 and 1 using Min-Max scaling.

2. **Denoising Autoencoder (Feature Extraction):**
   - An Autoencoder is designed with a series of `Conv2D` and `MaxPooling2D` layers to compress the image into a 256-dimensional latent feature vector.
   - Data Augmentation (Rotation and Zoom) is applied *only* to the inputs during training, forcing the autoencoder to act as a **Denoising Autoencoder**, making the extracted features more robust against variations.

3. **Classification:**
   - The 256-dimensional feature vectors are extracted from the encoder.
   - Due to potential class imbalances, the training features are balanced using `RandomOverSampler`.
   - The balanced features are fed into two distinct classifiers for comparison:
     1. **Random Forest Classifier**: A robust classical ML algorithm.
     2. **Deep Neural Network (MLP)**: A dense neural network with Dropout regularization to prevent overfitting.
   - A Learning Rate Scheduler is included to help find the optimal learning rate for the neural network.

## How to Run

1. **Environment:** You can run this project in a Jupyter Notebook environment (e.g., local Jupyter, Kaggle, Google Colab).
2. **Dataset:** Ensure the path to the MosMedData dataset in the `dataset_path` variable is correctly pointing to your local or cloud storage.
3. **Execution:** Run the cells sequentially from top to bottom.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Scikit-Learn
- Nibabel
- Imbalanced-Learn (imblearn)
- Numpy, Matplotlib, Seaborn, Tqdm
