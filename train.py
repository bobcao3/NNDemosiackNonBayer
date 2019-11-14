from __future__ import print_function

import sys
from keras.preprocessing.image import ImageDataGenerator

from model import seg,unet
import numpy as np
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,TerminateOnNaN
from scipy import misc
from keras.callbacks import TensorBoard
import math
import cv2 as cv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.setrecursionlimit(10000)
import time

# dataset
sensor_size = 32
focal_length = 60

# data
path_root = 'D:/demosaicking/Flickr500/bayer_128/'
img_rows, img_cols, img_channels = 128, 128, 3
batch_size = 8
nb_epoch = 3000
train_samples = 22000
vali_samples = 3000

# model
dropout_rate = 0.0
model_name = 'unet'
learning_rate = 1e-3

# loss func


# Normalize
def Normalize(data, ceiling):
    max = data[0][0]
    min = data[0][0]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] >max: max =data[i][j]
            if data[i][j] <min: min =data[i][j]

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = (data[i][j]-min)/(max-min) * ceiling
    return data

def data_generator(str):
    if str =='train':
        epochs = 0
    elif str =='vali':
        epochs = train_samples/batch_size

    phase_path = path_root + 'x/Img'
    groundtruth_path = path_root + 'y/Img'

    while True:
        phase_list = []
        groundtruth_list = []
        for i in range(batch_size):

            phase_tempt = cv.imread(phase_path+np.str(int(batch_size*epochs + i)) + '.png', 0)
            phase_tempt = np.asarray(phase_tempt).astype(float)
            phase_tempt = phase_tempt / 255
            phase_list.append(phase_tempt)

            groundtruth_tempt = cv.imread(groundtruth_path + np.str(int(batch_size*epochs + i)) + '.png')
            groundtruth_tempt = groundtruth_tempt.astype(float)
            groundtruth_tempt = groundtruth_tempt/255
            groundtruth_list.append(groundtruth_tempt)

        phase_list = np.asarray(phase_list).astype(float)
        groundtruth_list = np.asarray(groundtruth_list).astype(float)

        epochs += 1
        if epochs >= train_samples/batch_size:
            if str == 'train':
                epochs = 0
            elif str == 'vali':
                epochs = train_samples / batch_size

        yield phase_list, groundtruth_list


def vali_data():
    x_list =[]
    y_list = []
    for i in range(train_samples, train_samples+vali_samples):
        x = cv.imread(path_root+'x/Img'+np.str(i)+'.png')
        x = x / 255
        x_list.append(x)

        y = cv.imread(path_root+'y/Img'+np.str(i)+'.png')
        y = y/255
        y_list.append(y)

    x_list = np.asarray(x_list).astype(float)
    y_list = np.asarray(y_list).astype(float)

    return x_list, y_list


def create_model(str):

    model = unet.unet(img_rows,img_cols,img_channels)
    print("Model created")
    losses = "logcosh"
    # model.summary()

    optimizer = Adam(lr=learning_rate) # Using Adam instead of SGD to speed up training
    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=["mse"])
    print("Finished compiling")
    print("Building model...")
    return model


def callback():

    lr_reducer = ReduceLROnPlateau(monitor='loss', factor=np.sqrt(0.1),
                                cooldown=0, patience=10, min_lr=0.5e-6)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20)

    model_checkpoint = ModelCheckpoint(
        "weight/{}-{}-{}-{}-{}.h5".format(model_name, learning_rate, train_samples, img_rows,
                                          time.asctime(time.localtime(time.time()))),
        monitor="val_loss", save_best_only=True, save_weights_only=True)

    tb = TensorBoard(log_dir="logs/{}-{}-{}-{}-{}.h5".format(model_name, learning_rate, train_samples, img_rows,time.asctime(time.localtime(time.time()))),
                     histogram_freq=1,  # frequency for the histogram, 0 for not calculating
                     batch_size=batch_size,  # size of data to be used to calculate hist
                     write_graph=True,  # store network structure map
                     write_grads=False,  # virtualize grad hist
                     write_images=True,  # virtualize parameter
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)
    termonNan = TerminateOnNaN()

    callbacks=[lr_reducer, model_checkpoint, termonNan, tb]
    # callbacks = []
    print("callback func loaded")
    return callbacks


if __name__ == '__main__':

    model = create_model(model_name)
    (x, y) = next(data_generator('train'))

    history = model.fit_generator(data_generator('train'),
                                  steps_per_epoch=math.ceil(train_samples / batch_size), epochs=nb_epoch, verbose=2,   #
                                  callbacks=callback(),
                                  # class_weight=None,
                                  # max_queue_size=1,
                                  validation_data=vali_data(),
                                  # validation_data= vali_generator(),
                                  validation_steps=math.ceil(vali_samples/batch_size),
                                  # validation_freq=1,
                                  # workers=1, use_multiprocessing=False,
                                  # shuffle=True
                                  )




