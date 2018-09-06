from os.path import join
import cv2
import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import random
import keras.backend as K
from config_v2 import *
import keras


def split_train_val(csv_driving_data, test_size=0.2):
    """
    Splits the csv containing driving data into training and validation

    :param csv_driving_data: file path of Udacity csv driving data
    :return: train_split, validation_split
    """
    with open(csv_driving_data, 'r') as f:
        reader = csv.reader(f)
        #driving_data = [row for row in reader][1:]
        driving_data = [row for row in reader]

    train_data, val_data = train_test_split(driving_data, test_size=test_size, random_state=1)

    return train_data, val_data


def preprocess(frame_bgr, verbose=False):
    """
    Perform preprocessing steps on a single bgr frame.
    These inlcude: cropping, resizing, eventually converting to grayscale

    :param frame_bgr: input color frame in BGR format
    :param verbose: if true, open debugging visualization
    :return:
    """
    # set training images resized shape
    h, w = CONFIG['input_height'], CONFIG['input_width']

    # crop image (remove useless information)
    frame_cropped = frame_bgr[CONFIG['crop_height'], :, :]

    # resize image
    frame_resized = cv2.resize(frame_cropped, dsize=(w, h))

    # eventually change color space
    if CONFIG['input_channels'] == 1:
        frame_resized = np.expand_dims(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)[:, :, 0], 2)

    if verbose:
        plt.figure(1), plt.imshow(cv2.cvtColor(frame_bgr, code=cv2.COLOR_BGR2RGB))
        plt.figure(2), plt.imshow(cv2.cvtColor(frame_cropped, code=cv2.COLOR_BGR2RGB))
        plt.figure(3), plt.imshow(cv2.cvtColor(frame_resized, code=cv2.COLOR_BGR2RGB))
        plt.show()

    return frame_resized.astype('float32')


def load_data_batch(data, batchsize=CONFIG['batchsize'], data_dir='data', augment_data=True, bias=0.5):
    """
    Load a batch of driving data from the "data" list.
    A batch of data is constituted by a batch of frames of the training track as well as the corresponding
    steering directions.

    :param data: list of training data in the format provided by Udacity
    :param batchsize: number of elements in the batch
    :param data_dir: directory in which frames are stored
    :param augment_data: if True, perform data augmentation on training data
    :param bias: parameter for balancing ground truth distribution (which is biased towards steering=0)
    :return: X, Y which are the batch of input frames and steering angles respectively
    """
    # set training images resized shape
    h, w, c = CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels']

    # prepare output structures
    X = np.zeros(shape=(batchsize, h, w, c), dtype=np.float32)
    y_steer = np.zeros(shape=(batchsize,), dtype=np.float32)
    y_throttle = np.zeros(shape=(batchsize,), dtype=np.float32)

    # shuffle data
    shuffled_data = shuffle(data)

    loaded_elements = 0
    while loaded_elements < batchsize:

        #ct_path, lt_path, rt_path, steer, throttle, brake, speed = shuffled_data.pop()
        ct_path, steer, throttle, brake, speed, time_stamp, lap = shuffled_data.pop()

        # cast strings to float32
        steer = np.float32(steer)
        throttle = np.float32(throttle)

        # randomly choose which camera to use among (central, left, right)
        # in case the chosen camera is not the frontal one, adjust steer accordingly
        
        delta_correction = CONFIG['delta_correction']
        #camera = random.choice(['frontal', 'left', 'right'])
        camera = random.choice(['frontal'])
        if camera == 'frontal':
            frame = preprocess(cv2.imread(join(data_dir, ct_path.strip())))
            steer = steer
        elif camera == 'left':
            frame = preprocess(cv2.imread(join(data_dir, lt_path.strip())))
            steer = steer + delta_correction
        elif camera == 'right':
            frame = preprocess(cv2.imread(join(data_dir, rt_path.strip())))
            steer = steer - delta_correction
        
        if augment_data:

            # mirror images with chance=0.5
            if random.choice([True, False]):
                frame = frame[:, ::-1, :]
                steer *= -1.

            # perturb slightly steering direction
            steer += np.random.normal(loc=0, scale=CONFIG['augmentation_steer_sigma'])

            # if color images, randomly change brightness
            if CONFIG['input_channels'] == 3:
                frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2HSV)
                frame[:, :, 2] *= random.uniform(CONFIG['augmentation_value_min'], CONFIG['augmentation_value_max'])
                frame[:, :, 2] = np.clip(frame[:, :, 2], a_min=0, a_max=255)
                frame = cv2.cvtColor(frame, code=cv2.COLOR_HSV2BGR)

        # check that each element in the batch meet the condition
        steer_magnitude_thresh = np.random.rand()
        if (abs(steer) + bias) < steer_magnitude_thresh:
            pass  # discard this element
        else:
            X[loaded_elements] = frame
            y_steer[loaded_elements] = steer
            y_throttle[loaded_elements] = throttle
            loaded_elements += 1
        #y = np.concatenate(([y_steer],[y_throttle]),axis=0)
        #pdb.set_trace()
    if K.backend() == 'theano':
        X = X.transpose(0, 3, 1, 2)
    return X,y_steer
    #return X, y.T


def generate_data_batch(data, batchsize=CONFIG['batchsize'], data_dir='data', augment_data=True, bias=0.5):
    """
    Generator that indefinitely yield batches of training data

    :param data: list of training data in the format provided by Udacity
    :param batchsize: number of elements in the batch
    :param data_dir: directory in which frames are stored
    :param augment_data: if True, perform data augmentation on training data
    :param bias: parameter for balancing ground truth distribution (which is biased towards steering=0)
    :return: X, Y which are the batch of input frames and steering angles respectively
    """
    while True:

        X, y = load_data_batch(data, batchsize, data_dir, augment_data, bias)

        yield X, y
"""
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_list, batch_size,
                 data_dir='data',augment_data=True,bias=0.5, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.data_list = data_list
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of data
        data_list_temp = [self.data_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(data_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X,y = load_data_batch(list_IDs_temp, batchsize=CONFIG['batchsize'], data_dir='data', augment_data=True, bias=0.5)
        return X, y
"""

if __name__ == '__main__':

    # debugging purpose
    train_data, val_data = split_train_val(csv_driving_data='data/driving_log_20180902.csv')
    X,y = load_data_batch(train_data)



