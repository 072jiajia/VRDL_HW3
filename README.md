# VRDL Homework 3
Code for mAP 0.68340 solution of VRDL Homework 3

## Abstract
In this work, I use [Cascade Mask R-CNN](https://github.com/facebookresearch/detectron2/blob/master/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml) as my model<br>
In testing phase, I resize the shorter edge of the image to 2 different lengths 400 and 800<br>
And merge the two predictions by nms with threshold = 0.7 which is equal to RPN's nms threshold

## Reference
[GitHub](https://github.com/facebookresearch/detectron2) Detectron2<br>
[Paper](https://arxiv.org/pdf/1712.00726v1.pdf) Cascade R-CNN<br>
[Paper](https://arxiv.org/pdf/1803.08494.pdf) Group Normalization<br>
[Paper](https://arxiv.org/pdf/1703.06211.pdf) Deformable Convolutional Networks

## Hardware
The following specs were used to create the solutions.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
- 3x GeForce RTX 2080 Ti

## Reproducing Submission
To reproduct my submission, do the following steps:
1. [Installation](#installation)
2. [Prepare Data](#dataset-preparation)
3. [Download Pretrained models](#pretrained-models)
4. [Inference](#inference)

## Producing Your Own Submission
To produce your own submission, do the following steps:
1. [Installation](#installation)
2. [Prepare Data](#dataset-preparation)
3. [Train and Make Submission](#train-and-make-prediction)

## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is recommended.
```
virtualenv .
source bin/activate
python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
pip3 install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install opencv-python
pip3 install shapely
```

If your CUDA version is not 11.0<br>
You might need to change all "cu110" to your cuda version<br>
Or you can [install detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) and [torch](https://pytorch.org) by yourself

## Dataset Preparation
You need to download the data [here](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK) by yourself.<br>
Unzip it and put them into the same directory below
```
VRDL_HW3
  + test_images
  | + 2007_000629.jpg
  | + 2007_001175.jpg
  | + ...
  | + 2011_003146.jpg
  | + test.json
  + train_images
  | + 2007_000033.jpg
  | + 2007_000042.jpg
  | + ...
  | + 2011_003271.jpg
  | + pascal_train.json
  + config.yaml
  + inference.py
  + train.py
```

## Pretrained models
You can download pre-trained model that used for my submission from this [link](https://drive.google.com/file/d/1ZgZ9yRhuzfZK2pV0Zw9L2-opRMlsgTM2/view?usp=sharing)<br>
And put it into the following directory:
```
VRDL_HW3
  + test_images
  + train_images
  + bestmodel_HW3.pth
  + config.yaml
  + inference.py
  + train.py
```

## Inference
Run the following command to reproduct my prediction.
```
python3 inference.py bestmodel_HW3.pth
```
It will generate a file named prediction.json and it is my prediction whose mAP is 0.68340


## Train and Make Prediction
You can simply run the following command to train your own models and make submission.
```
$ python train.py
```

The expected training time is:

GPUs | MAX_ITER | Training Time
------------- | ------------- | ------------- 
3x 2080 Ti | 30000 | 11 hours

After finishing training your model, run the following command to make your prediction
```
python3 inference.py {your model's path}
```
It will generate a file prediction.json which is the prediction of the testing dataset<br>
Use this json file to make your submission!

## Citation
```
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
