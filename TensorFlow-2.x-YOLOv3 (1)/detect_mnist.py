#================================================================
#
#   File name   : detect_mnist.py
#   Author      : PyLessons
#   Created date: 2020-08-12
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : mnist object detection example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import *

gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus) > 0:
    for k in range(len(gpus)):
        tf.config.experimental.set_memory_growth(gpus[k], True)
        #tf.config.experimental.
        print('memory growth:', tf.config.experimental.get_memory_growth(gpus[k]))
else:
    print("Not enough GPU hardware devices available")

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use keras weights

while True:
    ID = random.randint(0, 25)
    #label_txt = "mnist/mnist_test.txt"
    #label_txt = "VOCdevkit/VOC2007/2007_test.txt"
    label_txt = "gj/gj_val.txt"
    image_info = open(label_txt).readlines()[ID].split()
    image_path = image_info[0]    
      #detect_image(yolo, image_path, "mnist_test.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    detect_image(yolo, image_path, "voc_test.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
