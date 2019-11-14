import numpy as np
import cv2 as cv
import math

width = 10000
height = 10000

'''
for i in range(1,501):
    if i<10:
        str = '00' + np.str(i)
    elif 10<=i<100:
        str = '0' + np.str(i)
    else:
        str = np.str(i)

    img = cv.imread('D:/demosaicking/Flickr500/Img'+str+'.png',0)
    img = np.asarray(img)
    print(img.shape)
    if width >= img.shape[1]:
        width = img.shape[1]
    if height >= img.shape[0]:
        height = img.shape[0]
'''
# print('width = '+np.str(width)+"  height = "+np.str(height))


def psnr(img, background):
    error = img - background
    square_error = np.multiply(error, error)

    mse = np.sum(square_error) / (img.shape[0] * img.shape[1])
    psnr = 20 * math.log10(max(img)) - 10 * math.log10(mse)

    return psnr
