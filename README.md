# ImageSegment with Unet3D
> This is an example of 3D ImageSegment

## Prerequisities
The following dependencies are needed:
- numpy >= 1.11.1
- SimpleITK >=1.0.1
- opencv-python >=3.3.0
- tensorflow-gpu ==1.8.0
- pandas >=0.20.1
- scikit-learn >= 0.17.1

## How to Use
(re)implemented the model with tensorflow in the paper of "Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., and Ronneberger, O. (2016) 3D U-net: Learning dense volumetric segmentation from sparse annotation. MICCAI 2016"

Unet3d structure is in model create_conv_net function

My Machine is GTX1080,when train it,the batchsize is one,too many batchsize lead to memory out.If you want sovel the bottleneck,you can change GTX1080 to GTX1080Ti,or take mutilGPU for training

## Contact
* https://github.com/junqiangchen
* email: 1207173174@qq.com
* WeChat Public number: 最新医学影像技术
