import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# import mpld3

# mpld3.enable_notebook()

# Read Images
import cv2
import numpy as np
import random



def im2cfa(im, cfa_filter):
    imshape = im.shape
    cfa = im.copy()
    filtershape = cfa_filter.shape
    for x in range(0, imshape[0]):
        for y in range(0, imshape[1]):
            rgb = cfa[x, y]
            f = cfa_filter[x % filtershape[0], y % filtershape[1]]
            cfa[x, y] = rgb * f
    return cfa


bayer_filter = np.array([   # Bayer:
    [[1, 0, 0], [0, 1, 0]], # R G
    [[0, 1, 0], [0, 0, 1]], # G B
])

'''
for i in range(1,500):
    if i<10:
        str = '00' + np.str(i)
    elif 10<=i<100:
        str = '0' + np.str(i)
    else:
        str = np.str(i)

    img = cv2.imread('D:/demosaicking/Flickr500/Img'+str+'.png')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = im2cfa(img,bayer_filter)
    cv2.imwrite('D:/demosaicking/Flickr500_bayer/Img'+str+'.png', cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    print(i)
'''

patch_per_img = 500
if __name__=='__main__':
        for i in range(1,501):
            if i < 10:
                str = '00' + np.str(i)
            elif 10 <= i < 100:
                str = '0' + np.str(i)
            else:
                str = np.str(i)

            img = cv2.imread('D:/demosaicking/Flickr500_bayer/Img'+str+'.png')
            img2 = cv2.imread('D:/demosaicking/Flickr500/Img' + str + '.png')

            # print('img.shape='+np.str(img.shape[0]))
            for j in range(patch_per_img):
                randrow = random.randint(0,img.shape[0]-128)
                randcol = random.randint(0,img.shape[1]-128)

                img_tempt = img[randrow:randrow+128,randcol:randcol+128]
                img2_tempt = img2[randrow:randrow+128,randcol:randcol+128]

                cv2.imwrite('D:/demosaicking/Flickr500_bayer_128/x/Img' + np.str(i*patch_per_img+j) + '.png', cv2.cvtColor(img_tempt, cv2.COLOR_RGB2BGR))
                cv2.imwrite('D:/demosaicking/Flickr500_bayer_128/y/Img' + np.str(i*patch_per_img+j) + '.png', cv2.cvtColor(img2_tempt, cv2.COLOR_RGB2BGR))
                print(i*patch_per_img+j)


