from __future__ import division
from unet3d.model import unet3dModule
import numpy as np
import pandas as pd


def train():
    '''
   Preprocessing for dataset
   '''
    # Read data set (Train data from CSV file)
    csv_data = pd.read_csv('train.csv')
    data = csv_data.iloc[:, :].values
    np.random.shuffle(data)

    # Extracting images and labels from given data
    data = data.astype(np.float)
    # Normalize from [0:255] => [0.0:1.0]
    data = np.multiply(data, 1.0 / 255.0)
    # For images
    image_size = 512 * 512 * 128
    images = data[:, image_size:]
    # For labels
    labels = data[:, :image_size - 1]
    # Split data into training & validation
    # Split data into training & validation
    train_images = images[0:]
    train_labels = labels[0:]

    unet3d = unet3dModule(128, 512, 512, 1)
    unet3d.train(train_images, train_labels, "E:\pythonworkspace\\neusoftProject\\NeusoftLibrary\\test\\model\\unet3d",
                 "E:\\pythonworkspace\\neusoftProject\\NeusoftLibrary\\test\\log", 0.0001, 0.8, 0.7, 5, 10)


def predict():
    test_imagesdata = pd.read_csv("test.csv")
    # For images
    image_size = 512 * 512 * 128
    test_images = test_imagesdata[:, image_size:]
    test_images = test_images.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)
    unet3d = unet3dModule(128, 512, 512, 1)
    predictvalue = unet3d.prediction("E:\pythonworkspace\\neusoftProject\\NeusoftLibrary\\test\\model\\unet3d",
                                     test_images)
    print(predictvalue[0])


def main(argv):
    if argv == 1:
        train()
    else:
        predict()


if __name__ == "__main__":
    main(2)
