# COVID-19 CT Scan Classification using Autoencoder Feature Extraction

This project involves the medical image processing of 3D CT scans (in NIfTI format) to classify COVID-19 severity. It utilizes TensorFlow/Keras for deep learning models and `nibabel` for loading the medical imaging data.

## Overview
The goal of this project is to accurately classify COVID-19 severity from chest CT scans using a two-step approach:
1.  **Feature Extraction:** Training a Convolutional Autoencoder to compress the images into a lower-dimensional latent space.
2.  **Classification:** Using the extracted features from the trained encoder to train classification models (Random Forest and Neural Networks) to predict the severity class.

## Dataset
The project is built to work with the **MosMedData Chest CT Scans with COVID-19 Related Findings** dataset.
The scans are categorized into 5 classes (CT-0 to CT-4), representing increasing severity. Due to a small sample size in class CT-4, it is merged with class CT-3.
The final classes used are:
*   CT-0 (Normal)
*   CT-1 (Mild)
*   CT-2 (Moderate)
*   CT-3 (Severe/Critical)

## Methodology

### 1. Data Preprocessing
*   **Loading:** 3D NIfTI files are loaded using `nibabel`.
*   **Slice Extraction:** To optimize processing, only the middle slice of the 3D scan is extracted, converting the 3D data into a 2D format.
*   **Normalization:** The 2D slice is normalized (Min-Max scaling).
*   **Resizing:** The slices are resized to a target dimension (e.g., 128x128).
*   **Data Augmentation:** Mild random rotation and zoom are applied during the autoencoder training to prevent overfitting.

### 2. Feature Extraction (Autoencoder)
*   A Convolutional Autoencoder is designed.
*   **Encoder:** Compresses the 128x128 image into a 256-dimensional latent representation.
*   **Decoder:** Reconstructs the image from the 256 features to the original size.
*   The autoencoder is trained using Mean Squared Error (MSE) loss.
*   After training, the encoder part is used to extract 256 features for all images.
*   Principal Component Analysis (PCA) is applied to visualize the features in a 2D space.

### 3. Classification
*   The extracted features suffer from class imbalance. Random Over Sampler is applied to balance the training data.
*   Two models are evaluated for classification:
    *   **Random Forest Classifier:** Used as a strong baseline model.
    *   **Neural Network Classifier:** A multi-layer perceptron (MLP) built with TensorFlow/Keras.
*   A Learning Rate Finder is implemented to identify the optimal learning rate before fully training the Neural Network classifier.
*   Comprehensive evaluation metrics are calculated, including Accuracy, Precision, Recall, F1-Score, ROC AUC, along with Confusion Matrices, ROC curves, and Precision-Recall curves.

## Requirements
The main libraries required for this project include:
*   `tensorflow` (and Keras)
*   `nibabel`
*   `scikit-learn`
*   `imblearn`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `tqdm`

## Notebook
The main code is located in the Jupyter Notebook: `autoencoder-featureext-01.ipynb`.
*   It includes markdown cells explaining each block of code.
*   Make sure to clear cell outputs when saving the notebook to prevent large file diffs.

---

# طبقه بندی اسکن سی‌تی کووید-۱۹ با استفاده از استخراج ویژگی‌های رمزگذار خودکار (Autoencoder)

این پروژه شامل پردازش تصاویر پزشکی از اسکن‌های سه‌بعدی سی‌تی (فرمت NIfTI) برای طبقه بندی شدت کووید-۱۹ است. از TensorFlow/Keras برای مدل‌های یادگیری عمیق و از `nibabel` برای بارگذاری داده‌های تصویربرداری پزشکی استفاده می‌کند.

## بررسی اجمالی
هدف از این پروژه، طبقه بندی دقیق شدت کووید-۱۹ از روی اسکن‌های سی‌تی قفسه سینه با استفاده از یک رویکرد دو مرحله‌ای است:
1.  **استخراج ویژگی:** آموزش یک رمزگذار خودکار پیچشی (Convolutional Autoencoder) برای فشرده‌سازی تصاویر به یک فضای پنهان با ابعاد پایین‌تر.
2.  **طبقه بندی:** استفاده از ویژگی‌های استخراج‌شده از رمزگذارِ آموزش‌دیده، برای آموزش مدل‌های طبقه‌بندی (جنگل تصادفی و شبکه‌های عصبی) جهت پیش‌بینی کلاس شدت بیماری.

## مجموعه داده (Dataset)
این پروژه برای کار با مجموعه داده **MosMedData Chest CT Scans with COVID-19 Related Findings** ساخته شده است.
اسکن‌ها به 5 کلاس (CT-0 تا CT-4) دسته‌بندی می‌شوند که نشان‌دهنده افزایش شدت بیماری است. به دلیل حجم کم نمونه‌ها در کلاس CT-4، این کلاس با کلاس CT-3 ادغام می‌شود.
کلاس‌های نهایی مورد استفاده عبارتند از:
*   CT-0 (طبیعی)
*   CT-1 (خفیف)
*   CT-2 (متوسط)
*   CT-3 (شدید/بحرانی)

## روش‌شناسی (Methodology)

### 1. پیش‌پردازش داده‌ها
*   **بارگذاری:** فایل‌های سه‌بعدی NIfTI با استفاده از `nibabel` بارگذاری می‌شوند.
*   **استخراج برش:** برای بهینه‌سازی پردازش، تنها برش میانی اسکن سه‌بعدی استخراج می‌شود و داده‌های سه‌بعدی به فرمت دوبعدی تبدیل می‌شوند. (همچنین از پنجره‌بندی واحد هانسفیلد - HU - برای پیش‌پردازش می‌توان استفاده کرد).
*   **نرمال‌سازی:** برش دوبعدی نرمال می‌شود (تغییر مقیاس Min-Max).
*   **تغییر اندازه:** برش‌ها به ابعاد هدف (مثلاً 128x128) تغییر اندازه داده می‌شوند.
*   **افزایش داده‌ها (Data Augmentation):** چرخش تصادفی ملایم و زوم در طول آموزش رمزگذار خودکار اعمال می‌شود تا از بیش‌برازش (Overfitting) جلوگیری شود.

### 2. استخراج ویژگی (Autoencoder)
*   یک رمزگذار خودکار پیچشی (Convolutional Autoencoder) طراحی شده است.
*   **رمزگذار (Encoder):** تصویر 128x128 را به یک نمایش پنهان با 256 ویژگی فشرده می‌کند.
*   **رمزگشا (Decoder):** تصویر را از 256 ویژگی به اندازه اصلی بازسازی می‌کند.
*   رمزگذار خودکار با استفاده از خطای میانگین مربعات (MSE) آموزش داده می‌شود.
*   پس از آموزش، بخش رمزگذار برای استخراج 256 ویژگی برای همه تصاویر استفاده می‌شود.
*   تحلیل مؤلفه‌های اصلی (PCA) برای تصویرسازی ویژگی‌ها در یک فضای دوبعدی اعمال می‌شود.

### 3. طبقه‌بندی (Classification)
*   ویژگی‌های استخراج‌شده از عدم تعادل کلاس (Class Imbalance) رنج می‌برند. برای متعادل کردن داده‌های آموزشی از Random Over Sampler استفاده می‌شود.
*   دو مدل برای طبقه بندی ارزیابی می‌شوند:
    *   **طبقه‌بند جنگل تصادفی (Random Forest):** به عنوان یک مدل پایه قوی استفاده می‌شود.
    *   **طبقه‌بند شبکه عصبی:** یک پرسپترون چند لایه (MLP) که با TensorFlow/Keras ساخته شده است.
*   یک یابنده نرخ یادگیری (Learning Rate Finder) برای شناسایی نرخ یادگیری بهینه پیش از آموزش کامل شبکه عصبی پیاده‌سازی شده است.
*   معیارهای ارزیابی جامع محاسبه می‌شوند، از جمله: دقت (Accuracy)، Precision، Recall، F1-Score، ROC AUC، به همراه ماتریس‌های درهم‌ریختگی (Confusion Matrix)، منحنی‌های ROC و منحنی‌های Precision-Recall.

## پیش‌نیازها
کتابخانه‌های اصلی مورد نیاز برای این پروژه عبارتند از:
*   `tensorflow` (و Keras)
*   `nibabel`
*   `scikit-learn`
*   `imblearn`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `tqdm`

## دفترچه راهنما (Notebook)
کد اصلی در Jupyter Notebook به آدرس `autoencoder-featureext-01.ipynb` قرار دارد.
*   این فایل شامل سلول‌های نشانه گذاری (Markdown) است که هر بلوک کد را توضیح می‌دهد.
*   هنگام ذخیره نوت‌بوک، مطمئن شوید که خروجی سلول‌ها پاک شده‌اند تا از ایجاد فایل‌های با حجم بالا جلوگیری شود.