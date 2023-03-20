# DL_HW1_Image_Classification_Pipeline
This project aims to build an "Image Classification Pipeline" based on the following:

- [x] Reading images to an array
- [x] Feature extraction (transform the image into a fixed-length feature vector) - HOG
- [x] Apply XGBoost, Catboost and LightGBM classifier to verify the performance

[Click to read the code](https://github.com/podo47/DL_HW1_Image_Classification_Pipeline/blob/a55acafb95d2cccb61aa81e0fa56a99fe9745e02/DL__hw1.ipynb)

Due to the limitation of computing resource on free colab version, this project only chooses a rough way to process the data and training. If your resourse and equipments are good enough, base on this project's pipeline, you can also modify the parameters of classifier and HOG, or increase images' size (higher spatial resolution and more detailed information) for more precise results.

--------------

## 1. Data
### Raw data
* Image 

  * Please go to [here](https://drive.google.com/uc?id=1kwYYWL67O0Dcbx3dvZIfbGg9NiHdyisr&export=download) to download the images in your computer.

  * Or just use following code to download and unzip the image on colab. Remember to put them under the folder ```./images/.```

   ``` python
   !pip install googleDriveFileDownloader 
   ```
   ``` python
   from googleDriveFileDownloader import googleDriveFileDownloader
   gdownloader = googleDriveFileDownloader()
   gdownloader.downloadFile("https://drive.google.com/uc?id=1kwYYWL67O0Dcbx3dvZIfbGg9NiHdyisr&export=download") 
   ```
   ``` python
   !unzip /content/images.zip -d /content/drive/MyDrive
   ```
  
 * There are 50 categories in total, with 48 categories having 1300 images each, except for two categories with 755 and 1070 images respectively.   There are 64225 images in total.
 
* Txt file containing locations of train, validation and test data : ```./train.txt``` ```./val.txt``` ```./test.txt```

### The processed data (extracted image features) - [hog_data](https://github.com/podo47/DL_HW1_Image_Classification_Pipeline/blob/0869ecab2bdd3524dda8420b8c17ef11e646e1b5/hog_data.npz)

Since it cost much time to generate feature matrices, they have been produced in advance and placed in the file ```./hog_data.npz```

If you want to do the process of feature extraction by yourself, just do all the ```Part 1 : Data preprocessing``` then you will get the same ```./hog_data.npz``` file.


## 2. Usage
### Part 1 : Data preprocessing
 
 * Import libraries
   ```python
   import os
   import numpy as np # linear algebra
   import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
   import matplotlib.pyplot as plt # for data visualization purposes
   import seaborn as sns # for statistical data visualization
   from skimage.feature import hog
   from skimage.color import rgb2gray
   from PIL import Image
   import cv2
   ```
   ```
   import warnings
   warnings.filterwarnings('ignore')
   ```
 * Import data
   ```python
   train = pd.read_csv('/content/drive/MyDrive/images/train.txt', sep=" ",header=None)
   train_dir = np.array(train[0])
   train_y = np.array(train[1])

   valid = pd.read_csv('/content/drive/MyDrive/images/val.txt', sep=" ",header=None)
   valid_dir = np.array(valid[0])
   valid_y = np.array(valid[1])

   test = pd.read_csv('/content/drive/MyDrive/images/test.txt', sep=" ",header=None)
   test_dir = np.array(test[0])
   test_y = np.array(test[1])
   ```

 * Load feature vector and target variable
 
    🚩 Prepared data :  [hog_data.npz](https://github.com/podo47/DL_HW1_Image_Classification_Pipeline/blob/0869ecab2bdd3524dda8420b8c17ef11e646e1b5/hog_data.npz)
 
   ```python
   with np.load("/content/drive/MyDrive/images/hog_data.npz") as data:
     hog_train = data["hog_train_data"]
     hog_valid = data["hog_valid_data"]
     hog_test = data["hog_test_data"]
   ```
   
### Part 2: Classifier

 * XGBoost
   
   🚩 Prepared data :  [xgb_model.dat](https://github.com/podo47/DL_HW1_Image_Classification_Pipeline/blob/a55acafb95d2cccb61aa81e0fa56a99fe9745e02/xgb_model.dat)
   
   Load the pretrained model
   ```python
   xgb_model = joblib.load(xgb_file)
   ```
 * Catboost

   🚩 Prepared data :  [cbc_model.dat](https://github.com/podo47/DL_HW1_Image_Classification_Pipeline/blob/a55acafb95d2cccb61aa81e0fa56a99fe9745e02/cbc_model.dat) , [eval_results_cbc.pkl](https://github.com/podo47/DL_HW1_Image_Classification_Pipeline/blob/a55acafb95d2cccb61aa81e0fa56a99fe9745e02/eval_results_cbc.pkl)
   
   * Load the pretrained model
   ```python
   cbc_model = joblib.load(cbc_file)
   ```

   * Load the pretrained model's training and validation metrics for ploting the training and validation curves
   ```python
   with open(path_cbc, 'rb') as f:
    eval_results_cbc = joblib.load(f)
    ```

 * Light GBM

   🚩 Prepared data :  [lgbm_model.dat](https://github.com/podo47/DL_HW1_Image_Classification_Pipeline/blob/a55acafb95d2cccb61aa81e0fa56a99fe9745e02/lgbm_model.dat)
   
   Load the pretrained model
   ```python
   lgbm_model = joblib.load(lgbm_file)
   ```
### Part 3: Performance

 * Comparison of model's accuracy on validation data
 * Comparison of model's accuracy on test data

## 3. Reference
 
 1. HOG https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
 2. ChatGPT
 
 
  
 


