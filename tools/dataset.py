import img2raw as im
import sys
import os
import math
import numpy as np

import cv2

import matplotlib as plt
import random

if (len(sys.argv) != 4):
    print("Usage: python3 dataset.py orig_dir output_dir camera_model")
    sys.exit(-1)

orignals_dir = sys.argv[1]
output_dir_groundtruth = os.path.join(sys.argv[2], "ref")
output_dir_raw = os.path.join(sys.argv[2], "raw")
camera = sys.argv[3]

from pathlib import Path
path = Path(output_dir_groundtruth)
path.mkdir(parents=True, exist_ok=True)
path = Path(output_dir_raw)
path.mkdir(parents=True, exist_ok=True)

def augmentAndWrite(filename, oimg, index):
    index2 = 0
    for i in range(0, 4):
        img = oimg * random.uniform(0.5, 2.0) # -1 EV to + 1 EV

        # Write reference data
        ofile_bin = os.path.join(output_dir_groundtruth, filename + "." + str(index) + "." + str(i) + ".bin")
        print("Output: ", ofile_bin)
        ofile_png = os.path.join(output_dir_groundtruth, filename + "." + str(index) + "." + str(i) + ".jpg")
        cv2.imwrite(ofile_png, cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        im.writeBinImg(img, ofile_bin)
        # Write raw data
        ofile_bin = os.path.join(output_dir_raw, filename + "." + str(index) + "." + str(i) + ".bin")
        cfa = im.im2cfa(img, im.cfa2rgb(im.bayer_cfa0, camera))
        # cfa = im.cfaAddNoise(cfa)
        im.writeBinImg(cfa, ofile_bin)

        oimg = cv2.rotate(oimg, cv2.ROTATE_90_CLOCKWISE)

from multiprocessing import Pool

def processFile(filename):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        fullpath_orig = os.path.join(orignals_dir, filename)
        image_orig = im.readImg(fullpath_orig)
        # Downscale by 4 to reduce effects of image compression, camera noise, etc
        size_orig = (int(image_orig.shape[1] / 4), int(image_orig.shape[0] / 4))
        image_orig = cv2.resize(image_orig, size_orig, interpolation = cv2.INTER_AREA)
        # Slice images to 128x128 patches
        index = 0
        print("Processing ", fullpath_orig)
        for i in range(0, math.floor(size_orig[1] / 128)):
            for j in range(0, math.floor(size_orig[0] / 128)):
                image = image_orig[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128, :]
                augmentAndWrite(filename, image, index)
                index += 1
  
if __name__ == '__main__':
    with Pool(12) as p:
        p.map(processFile, os.listdir(orignals_dir))
    print("Renaming files")
    index = 0
    for filename in os.listdir(output_dir_groundtruth):
        if filename.endswith(".jpg"):
            os.rename(os.path.join(output_dir_groundtruth, filename), os.path.join(output_dir_groundtruth, str(index) + ".jpg"))
        if filename.endswith(".bin"):
            os.rename(os.path.join(output_dir_groundtruth, filename), os.path.join(output_dir_groundtruth, str(index) + ".bin"))
            os.rename(os.path.join(output_dir_raw, filename), os.path.join(output_dir_raw, str(index) + ".bin"))
            index += 1
