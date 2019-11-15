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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.setrecursionlimit(10000)
import datetime

import tools.img2raw as im

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# dataset
sensor_size = 32
focal_length = 60

if len(sys.argv) != 2:
    print("Usage: python3 train.py path_to_dataset")
    sys.exit(-1)

# data
path_root = sys.argv[1]
img_rows, img_cols, img_channels = 128, 128, 3
batch_size = 16
nb_epoch = 3000
train_samples = 75000
vali_samples = 3000

indices = list(range(0, train_samples + vali_samples))
np.random.shuffle(indices)

phase_path = os.path.join(path_root, 'raw')
groundtruth_path = os.path.join(path_root, 'ref')

# model
dropout_rate = 0.0
model_name = 'unet'
learning_rate = 2e-4

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

    while True:
        phase_list = []
        groundtruth_list = []
        for i in range(batch_size):

            phase_tempt = im.readBinImg(os.path.join(phase_path, np.str(indices[batch_size*epochs + i]) + '.bin'))
            phase_list.append(phase_tempt)

            groundtruth_tempt = im.readBinImg(os.path.join(groundtruth_path, np.str(indices[batch_size*epochs + i]) + '.bin'))
            groundtruth_list.append(groundtruth_tempt)

        phase_list = np.asarray(phase_list)
        groundtruth_list = np.asarray(groundtruth_list)

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
        x = im.readBinImg(os.path.join(phase_path, np.str(indices[i]) + '.bin'))
        x_list.append(x)

        y = im.readBinImg(os.path.join(groundtruth_path, np.str(indices[i]) + '.bin'))
        y_list.append(y)

    x_list = np.asarray(x_list)
    y_list = np.asarray(y_list)

    return x_list, y_list


def create_model(str):

    model = unet.unet(img_rows,img_cols,img_channels)
    print("Model created")
    losses = "logcosh"

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
        os.path.join("weight", "{}-{}-{}-{}-{}.h5".format(model_name, learning_rate, train_samples, img_rows,
                                          datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))),
        monitor="val_loss", save_best_only=True, save_weights_only=True)

    tb = TensorBoard(log_dir=os.path.join("logs", "{}-{}-{}-{}-{}.h5".format(model_name, learning_rate, train_samples, img_rows, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))),
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
                                  shuffle=True,
                                  validation_data=vali_data(),
                                  # validation_data= vali_generator(),
                                  validation_steps=math.ceil(vali_samples/batch_size),
                                  # validation_freq=1,
                                  # workers=1, use_multiprocessing=False,
                                  # shuffle=True
                                  )




