import matplotlib.pyplot as mp
import os
import numpy as np
import pca

def compress_images(DATA,k):
    Z = pca.compute_Z(DATA)
    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, k, 0)
    sliced_PCS = PCS[:, :k]
    UT = np.transpose(sliced_PCS)
    X_compressed = np.matmul(Z_star, UT)
    num_rows, num_cols = X_compressed.shape
    num_images = num_cols
    path = os.getcwd() + '/Output/'
    dir_exists = os.path.isdir(path)
    if not dir_exists:
        os.mkdir('Output/')
    i = 0
    while i < num_images:
        fname = path + 'compressed' + str(i) + '.jpg'
        image = X_compressed[:, i]
        min = np.amin(image)
        max = np.amax(image)
        rescaled_image = ((image - min)/(max - min)) * 255
        rescaled_image.shape = (60, 48)
        mp.imsave(fname, rescaled_image, cmap='gray')
        i += 1
    return 

def load_data(input_dir):
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
    return DATA