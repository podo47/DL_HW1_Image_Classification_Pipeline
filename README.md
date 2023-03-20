# DL_HW1_Image_Classification_Pipeline
This project aims to build an "Image Classification Pipeline" based on the following:

- [x] Reading images to an array
- [x] Feature extraction (transform the image into a fixed-length feature vector) - HOG
- [x] Apply XGBoost, Catboost and LightGBM classifier to verify the performance

[Click to read the code](https://github.com/podo47/DL_HW1_Image_Classification_Pipeline/blob/4a5784f62a7516388984875a0518e0378584dedd/DL__hw1.ipynb)
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
  
 * There are 50 categories in total, with 48 categories having 1300 images each, except for two categories with 755 and 1070 images respectively.      There are 64225 images in total.
 
* Txt file containing locations of train, validation and test data : ```./train.txt``` ```./val.txt``` ```./test.txt```

### The processed data (extracted image features)

Since it cost much time to generate feature matrices, they have been produced in advance and placed in the file ```./hog_data.npz```

[hog_data]()
