from modelTF import ModuelTF
import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage import data_dir, io, color, transform


def recognize_image(modelTF):
    while True:
        cwd = input('Input image filename:')
        try:
            img = Image.open(cwd).convert("RGB")
        except:
            print('Open Error! Try again!')
            continue
        else:
            label = modelTF.get_result(img)
            print(label)

    modelTF.close_session()


modelTF = ModuelTF()
recognize_image(modelTF)
