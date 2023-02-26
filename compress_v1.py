import matplotlib.pyplot as mp
import os, os.path
import numpy as np
import pca

def compress_images(DATA,k):
    return

def load_data(input_dir):
    # print(len([name for name in os.listdir(input_dir) if os.path.isfile(name)]))
    num_pics = 0
    for entry in os.scandir(input_dir):
        num_pics += 1
    DATA = np.zeros((2880, num_pics))
    i = 0
    for entry in os.scandir(input_dir):
        image = mp.imread(entry.path)
        flat_image = image.flatten()
        DATA[:, i] = flat_image
        i += 1
    print (DATA)
    return DATA