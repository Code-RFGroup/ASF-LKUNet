from tkinter import image_names
from PIL import Image
import os
import scipy.io as scio
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy.ndimage.filters import gaussian_filter
import SimpleITK as sitk
import torch



def extract_ordered_pathches(img,patch_size,stride_size,pad_way = 'zero'):

    h,w,c = img.shape
    patch_h,patch_w = patch_size
    stride_h,stride_w = stride_size
    left_h,left_w = (h - patch_h) % stride_h,(w-patch_w) % stride_w
    pad_h ,pad_w = (stride_h - left_h) % stride_h,(stride_w - left_w) %stride_w
    #assert(h - patch_h) % stride_h == 0 and (w - patch_w) % stride_w == 0
    h, w, c = img.shape

    n_pathes_y = (h + pad_h - patch_h) // stride_h + 1
    n_pathes_x = (w + pad_w - patch_w) // stride_w + 1
    n_pathches_per_img = n_pathes_y * n_pathes_x
    n_pathches = n_pathches_per_img
    patches = np.zeros((n_pathches,patch_h,patch_w,c),dtype=img.dtype)
    patch_idx = 0

    for i in range(n_pathes_y):
        for j in range(n_pathes_x):
            x1 = i * stride_h
            x2 = x1 + patch_h
            y1 = j * stride_w
            y2 = y1 + patch_w
            # print(x1, x2,y1,y2)
            if x2 >= h and y2 >= w:
                return_x1 = x1 - pad_h
                return_x2 = return_x1 + patch_h
                return_y1 = y1 - pad_w
                return_y2 = return_y1 + patch_w
                # print('xy', return_y1, return_y2)
                patches[patch_idx] = img[return_x1:return_x2, return_y1:return_y2]
            elif x2 >= h:
                return_x1 = x1 - pad_h
                return_x2 = return_x1 + patch_h
                # print('x',return_x1,return_x2)
                patches[patch_idx] = img[return_x1:return_x2, y1:y2]
            elif y2 >= w:
                return_y1 = y1 - pad_w
                return_y2 = return_y1 + patch_w
                # print('y',return_y1, return_y2)
                patches[patch_idx] = img[x1:x2, return_y1:return_y2]
            else:
                patches[patch_idx] = img[x1:x2, y1:y2]
            #print(x1, x2, y1, y2)
            patch_idx += 1
    return patches
