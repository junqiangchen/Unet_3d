from __future__ import division
from unet3d.model import unet3dModule
import numpy as np
import pandas as pd


def train():
    '''
   Preprocessing for dataset
   '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('mask.csv')
    csvimagedata = pd.read_csv('image.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]
    # Extracting images and labels from given data
    imagedata = imagedata.astype(np.float)
    maskdata = maskdata.astype(np.float)
    # Normalize from [0:255] => [0.0:1.0]
    images = np.multiply(imagedata, 1.0 / 255.0)
    labels = np.multiply(maskdata, 1.0 / 255.0)
    # Split data into training & validation
    train_images = images[0:]
    train_labels = labels[0:]

    unet3d = unet3dModule(3, 512, 512, 1)
    unet3d.train(train_images, train_labels, "E:\pythonworkspace\\test\\model\\unet3d",
                 "E:\\pythonworkspace\\test\\log", 0.0001, 0.8, 0.7, 50000, 1)


def predict():
    test_imagesdata = pd.read_csv("test.csv")
    # For images
    image_size = 512 * 512 * 128
    test_images = test_imagesdata[:, image_size:]
    test_images = test_images.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)
    unet3d = unet3dModule(128, 512, 512, 1)
    predictvalue = unet3d.prediction("E:\pythonworkspace\\test\\model\\unet3d",
                                     test_images)
    print(predictvalue[0])


def main(argv):
    if argv == 1:
        train()
    else:
        predict()


if __name__ == "__main__":
    main(2)
