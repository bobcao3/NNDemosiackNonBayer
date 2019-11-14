import keras
from keras.layers import * # Conv2D,Input,Lambda,Dropout, MaxPooling2D, Conv2DTranspose
from keras.models import *


def dmcnn(img_rows, img_cols, img_channels = 3):
    input = Input((img_rows, img_cols, img_channels))

    x = Conv2D(128,(9,9,3),kernel_initializer = 'Gaussian',activation='relu')(input)
    x = Conv2D(64,(1,1),kernel_initializer = 'Gaussian',activation='relu')(x)
    y = Conv2D(3,(5,5),kernel_initializer = 'Gaussian',activation='relu')(x)

    model = Model(inputs = input, outputs = y)

    return model


def dmcnnvd(img_rows, img_cols, img_channels):
    input = Input(img_rows, img_cols, img_channels)







