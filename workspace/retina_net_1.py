
# Experimenting with RetinaNet project
# https://github.com/fizyr/keras-retinanet

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
# from pylab import rcParam
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
import urllib
import os
import csv
import cv2
import time
from pathlib import Path
from PIL import Image

# import keras_retinanet models and utilities
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

pd.set_option('display.max_columns', None)

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

# rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

####################
## PRE-PROCESSING ##
####################
TRAIN_DATA = Path.home() / "PycharmProjects" / "retina_net" / "data" / "train"
TEST_DATA = Path.home() / "PycharmProjects" / "retina_net" / "data" / "test"
LABELS = Path.home() / "PycharmProjects" / "retina_net" / "data" / "train.csv"
train = pd.read_csv(LABELS)
# labels.head(2)
#     image_id  width  height                         bbox   source
# 0  b6ab77fd7   1024    1024   [834.0, 222.0, 56.0, 36.0]  usask_1
# 1  b6ab77fd7   1024    1024  [226.0, 548.0, 130.0, 58.0]  usask_1

# Need to format training data to match API input spec via 2 CSV files:
#   1) Annotations file:
#           path/to/image.jpg,x1,y1,x2,y2,class_name    (one per box)
#   2) Class mapping file:
#           class_name,id   (class name to ID mapping - one per box)

# Map given bounding box min coord and size to pure coordinates
bboxes = np.stack(train['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x_min', 'y_min', 'width', 'height']):
    train[column] = bboxes[:, i]

train["x_max"] = train.apply(lambda col: col.x_min + col.width, axis=1)
train["y_max"] = train.apply(lambda col: col.y_min + col.height, axis=1)
train.drop(columns=['bbox'], inplace=True)

# Checking freq dist of bounding boxes per image
train['image_id'].value_counts()

# Range checking on bounding box coordinates
train[train['x_min'] > 0]
train[train['y_min'] > 0]
train[train['x_max'] < 1024]
train[train['y_max'] < 1024]

# Add a label for all images (since all contain wheat)
# Useful for when/if we choose to include photos without wheat
train["class"] = "1"

# Append ".jpg" to image_id for ease of processing later on
train["image_id"] = train["image_id"].apply(lambda x: str(x) + ".jpg")
train["image_id"] = train["image_id"].apply(lambda x: Path.joinpath(TRAIN_DATA, str(x)))
train["image_id"] = train["image_id"].astype("str")
train.to_csv("wheat.csv", index=False)


################
## LOAD MODEL ##
################
#   1) config files?
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = Path.cwd() / "snapshots" / "resnet50_coco_best_v2.1.0.h5"
# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')





