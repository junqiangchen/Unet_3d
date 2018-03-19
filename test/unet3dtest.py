from __future__ import division
from unet3d.model import unet3dModule
import numpy as np
import pandas as pd
import cv2


def train():
    '''
   Preprocessing for dataset
   '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('D:\Project\python\\3D_luna16-lung-cancer\data/mask3d_lidc.csv')
    csvimagedata = pd.read_csv('D:\Project\python\\3D_luna16-lung-cancer\data/image3d_lidc.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    train_images = imagedata[perm]
    train_labels = maskdata[perm]

    unet3d = unet3dModule(3, 512, 512, 1)
    unet3d.train(train_images, train_labels, "D:\Project\python\download_projects\\Unet_3d-master\\test\\unet3d",
                 "D:\Project\python\download_projects\\Unet_3d-master\\test\\log", learning_rate=0.0001,
                 dropout_conv=0.8, train_epochs=5,
                 batch_size=1)


def predict():
    image1 = cv2.imread("D:\Project\python\download_projects\\Unet_3d-master\\1.bmp", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("D:\Project\python\download_projects\\Unet_3d-master\\2.bmp", cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread("D:\Project\python\download_projects\\Unet_3d-master\\3.bmp", cv2.IMREAD_GRAYSCALE)
    testimage = np.empty((3, 512, 512, 1))
    testimage[0, ...] = np.reshape(image1, (512, 512, 1))
    testimage[1, ...] = np.reshape(image2, (512, 512, 1))
    testimage[2, ...] = np.reshape(image3, (512, 512, 1))
    test_images = testimage.astype(np.float)
    # convert from [0:255] => [0.0:1.0]
    test_images = np.multiply(test_images, 1.0 / 255.0)
    unet3d = unet3dModule(3, 512, 512, 1)
    predictvalue = unet3d.prediction("D:\Project\python\download_projects\\Unet_3d-master\\test\\unet3d",
                                     test_images)
    cv2.imwrite("mask1.bmp", predictvalue[0])
    cv2.imwrite("mask2.bmp", predictvalue[1])
    cv2.imwrite("mask3.bmp", predictvalue[2])


def main(argv):
    if argv == 1:
        train()
    else:
        predict()


if __name__ == "__main__":
    main(2)
