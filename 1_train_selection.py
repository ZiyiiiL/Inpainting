import glob
import logging
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from skimage.feature import canny
from skimage.transform import rescale, resize
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer
import warnings
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, ConcatDataset

# 删除occluded pixel分布太均匀、即使总ratio低但难以截取256*256的patch的图像

def random_crop(img_train_split, nodata_mask_train_split, size=256, image_path=None):
    h, w = img_train_split.shape
    try_count = 0
    while try_count < (1e4 + 2):
        # 从train split中randomcrop出256*256大小的patch
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        img_patch = img_train_split[top:top + size, left:left + size]
        nodata_mask_patch = nodata_mask_train_split[top:top + size, left:left + size]
        if np.sum(nodata_mask_patch) == 0:
            return img_patch, nodata_mask_patch, try_count
        try_count += 1
        if try_count > 1e4:
            warnings.warn(
                f"Exceeded maximum attempt limit for current item. Please check {image_path} and select another ground truth image.",
                UserWarning)
            return img_patch, nodata_mask_patch, try_count

indir = '/scratch2/ziyliu/LAMA/lama/sate_dataset/train'
tiles = [tile.name for tile in Path(indir).iterdir() if tile.is_dir()]
pair_filenames = {}
iter_i = 0
for tile in tiles:
    if tile not in pair_filenames:
        pair_filenames[tile] = {"image_path": []}
    for image in os.listdir(indir + '/' + tile + '/image/'):
        if image.endswith('.tif'):
            image_path = indir + '/' + tile + '/image/' + image
            pair_filenames[tile]["image_path"].append(image_path)

for tile in tiles:
    for img_path in tqdm(pair_filenames[tile]["image_path"]):
        with rasterio.open(img_path) as src_img:
            img_array = src_img.read(1).astype('float32')
            nodata_mask = (img_array == (2 ** 16 - 1))
            mu = np.mean(img_array[img_array < 65535]) # 2048
            std = np.std(img_array[img_array < 65535]) # 2048
            min_value = np.maximum(0, mu - 3 * std)
            max_value = mu + 3 * std
            img_array = np.clip((img_array - min_value) / (max_value - min_value), 0, 1).astype('float32')
            img = img_array

        per = 0.7
        h, w = img.shape
        h_split = int(h * per)
        # take the upper 70% rows from the image as train split
        img_train_split = img[:h_split, :]
        nodata_mask_train_split = nodata_mask[:h_split, :]
        img_patch, nodata_mask_patch, try_count = random_crop(img_train_split, 
                                                                    nodata_mask_train_split, size=256, image_path=img_path)