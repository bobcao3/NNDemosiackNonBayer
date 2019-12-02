from __future__ import print_function

import sys
from keras.preprocessing.image import ImageDataGenerator

from model import seg,unet,DMCNN,HDRDMCNN
import numpy as np
from keras.utils import np_utils
from keras.optimizers import Adam, SGD
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

# dataset
sensor_size = 32
focal_length = 60

if len(sys.argv) != 2:
    print("Usage: python3 train.py path_to_dataset")
    sys.exit(-1)

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# data
path_root = sys.argv[1]
img_rows, img_cols, img_channels = 128, 128, 3
batch_size = 25 # DMCNN-VD: 25 UNET: 50 DMCNN: 100
nb_epoch = 1080
train_samples = 27000
vali_samples = 3000

# model
dropout_rate = 0.0
model_name = 'HDRDMCNN-HDR-Data'
learning_rate = 3e-4 # DMCNN-VD: 1e-4 UNET: 2e-4, DMCNN: 1e-3

indices = list(range(0, train_samples + vali_samples))
np.random.shuffle(indices)

phase_path = os.path.join(path_root, 'raw')
groundtruth_path = os.path.join(path_root, 'ref')

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


from keras import backend as K

def ssim_loss(y_true, y_pred):
    alpha = 0.84
    beta = 1.0
    L1 = K.mean(K.abs(y_true - y_pred))
    ms_ssim = 1.0 - tf.image.ssim_multiscale(y_true, y_pred, 2.0, filter_size=7)
    return alpha * ms_ssim + (1.0 - alpha) * L1 * beta

def create_model(str):

    model = HDRDMCNN.HDRdmcnn(img_rows,img_cols,img_channels)
    print("Model created")

    optimizer = Adam(lr=learning_rate) # Using Adam instead of SGD to speed up training
    model.compile(loss=ssim_loss, optimizer=optimizer, metrics=["mse"])
    print("Finished compiling")
    print("Building model...")
    return model

def callback():

    lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5,
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

    callbacks=[lr_reducer, model_checkpoint, termonNan, tb, early_stopper]
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
                                  # shuffle=True,
                                  validation_data=vali_data(),
                                  # validation_data= vali_generator(),
                                  validation_steps=math.ceil(vali_samples/batch_size),
                                  # validation_freq=1,
                                  # workers=1, use_multiprocessing=False,
                                  # shuffle=True
                                  )




