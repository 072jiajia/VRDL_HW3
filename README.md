# VRDL Homework 3
Code for mAP 0.68340 solution in VRDL Homework 3

## Abstract
In this work, I use Cascade Mask R-CNN as my model<br>

## Reference
Detectron2 [GitHub](https://github.com/facebookresearch/detectron2)<br>
Cascade R-CNN [Paper](https://arxiv.org/pdf/1712.00726v1.pdf)

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
pip3 install -r requirements.txt
```

## Dataset Preparation
You need to download the data [here](https://drive.google.com/drive/u/1/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl) by yourself.<br>
Unzip it and put them into the same directory below
```
VRDL_HW2
  + test_images
  | + 2007_000629.jpg
  | + 2007_001175.jpg
  | + ...
  | + 2011_003146.jpg
  + train_images
  | + 2007_000033.jpg
  | + 2007_000042.jpg
  | + ...
  | + 2011_003271.jpg
  + config.yaml
  + inference.py
  + train.py
```

## Pretrained models
You can download pre-trained model that used for my submission from this [link]({link})<br>
And put it into the following directory:
```
VRDL_HW2
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
python3 predict.py {your model's name}
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
