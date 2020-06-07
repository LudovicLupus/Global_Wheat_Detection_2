#!/bin/bash

## Run updates
sudo apt-get update
sudo apt-get upgrade

## Prepare project folder
mkdir kaggle
cd kaggle

## Clone personal repo
git clone https://github.com/LudovicLupus/Global_Wheat_Detection_2.git

## Change directory name to match local
mv mv Global_Wheat_Detection_2/ retina_net
cd retina_net
mkdir workspace/snapshots   # For storing MODEL files

## Clone Keras-RetinaNet repo as submodule
git clone --recurse-submodules https://github.com/fizyr/keras-retinanet.git
cd keras-retinanet

## Install package
pip intall .
python setup.py build_ext --inplace   # Might not be necessary

## Build pipenv virtual environment and install dependencies
cd ~/kaggle/retina_net
pip install pipenv
pipenv install
## Install tensorflow-gpu
pipenv install tensorflow-gpu

################################################################
## Ensure that the DATA and MODEL have been transferred to    ##
## the E2C instance before running the below (assuming these  ##
## files were transferred to the top-level directory in EC2   ##
################################################################
cd
mkdir kaggle/retina_net/data
mv resnet50_coco_best_v2.1.0.h5 kaggle/retina_net/workspace/snapshots/resnet50_coco_best_v2.1.0.h5
mv global-wheat-detection.zip kaggle/retina_net/data/global-wheat-detection.zip
unzip kaggle/retina_net/data/global-wheat-detection.zip

# Enter pipenv virtual shell
cd ~/kaggle/retina_net/
pipenv shell

###################################################################
## Execute training script from top-level 'retina_net' directory ##
###################################################################
keras-retinanet/keras_retinanet/bin/train.py \
--freeze-backbone \
    --random-transform \
    --weights workspace/snapshots/resnet50_coco_best_v2.1.0.h5\
    --batch-size 8 \
    --steps 500 \
    --epochs 10 csv workspace/annotations_2.csv workspace/classes.csv


    