from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense, Dropout, Lambda, LeakyReLU, Concatenate, MaxPooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.backend as K
from keras.models import model_from_json
from config_v5 import *
from load_data_v5 import generate_data_batch, split_train_val, DataGenerator
import sys

import pdb

def get_nvidia_model(summary=True):
    """
    Get the keras Model corresponding to the NVIDIA architecture described in:
    Bojarski, Mariusz, et al. "End to end learning for self-driving cars."

    The paper describes the network architecture but doesn't go into details for some aspects.
    Input normalization, as well as ELU activations are just my personal implementation choice.

    :param summary: show model summary
    :return: keras Model of NVIDIA architecture
    """
    init = 'glorot_uniform'

    if K.backend() == 'theano':
        input_frame_x = Input(shape=(CONFIG['input_channels'], NVIDIA_H, NVIDIA_W))
        input_frame_upper = Input(shape=(3, NVIDIA_H, NVIDIA_W))
    else:
        input_frame_x = Input(shape=(NVIDIA_H, NVIDIA_W, CONFIG['input_channels']))
        input_frame_upper = Input(shape=(NVIDIA_H, NVIDIA_W, 3))

    # standardize input
    input_x = Lambda(lambda z: z / 127.5 - 1.)(input_frame_x)
    input_upper = Lambda(lambda z: z / 127.5 - 1.)(input_frame_upper)

    x = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), init=init)(input_x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), init=init)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), init=init)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', init=init)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', init=init)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    # Another parallel CNN
    y = Convolution2D(24, 3, 3, border_mode='valid', init=init, activation='relu')(input_upper)
    y = MaxPooling2D(pool_size=(2,2))(y)
    y = Dropout(0.2)(y)
    y = Convolution2D(24, 3, 3, border_mode='valid', init=init, activation='relu')(y)
    y = MaxPooling2D(pool_size=(2,2))(y)
    y = Dropout(0.2)(y)
    y = Flatten()(y)

    # Concat and FC together
    x = Concatenate()([x,y])

    x = Dense(100, init=init)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(50, init=init)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(25, init=init)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(10, init=init)(x)
    x = LeakyReLU()(x)
    steer_out = Dense(1, init=init, name="steer")(x)
    speed_out = Dense(1, init=init, name="speed")(x)
    traffic_sign_cat_out = Dense(4, init=init, activation='softmax', name="sign_cat")(x) # no sign, change to left lane, change to right lane, others

    model = Model(input=[input_frame_x,input_frame_upper], output=[steer_out,speed_out,traffic_sign_cat_out])

    if summary:
        model.summary()
    return model


if __name__ == '__main__':

    # split udacity csv data into training and validation
    train_data, val_data = split_train_val(csv_driving_data='data/driving_log-ts-pf-all.csv')

    # get network model and compile it (default Adam opt)
    nvidia_net = get_nvidia_model(summary=True)
    # Model retrain.
    if len(sys.argv) == 3:
        model_path = sys.argv[1]
        with open(model_path) as jfile:
            nvidia_net = model_from_json(jfile.read())
        weights_path = sys.argv[2]
        nvidia_net.load_weights(weights_path)
    #nvidia_net.compile(optimizer='adam', loss='mse')
    nvidia_net.compile(optimizer='rmsprop', loss=['mae','mae','categorical_crossentropy'], loss_weights=[0.5,0.5,0.3])

    # json dump of model architecture
    with open('logs/model.json', 'w') as f:
        f.write(nvidia_net.to_json())

    # define callbacks to save history and weights
    checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
    logger = CSVLogger(filename='logs/history.csv')

    # start the training
    nvidia_net.fit_generator(
                         generator=DataGenerator(train_data,CONFIG['batchsize']),
                         steps_per_epoch = 1*(len(train_data)/CONFIG['batchsize']+1),
                         epochs=50,
                         validation_data=DataGenerator(val_data,CONFIG['batchsize'], augment_data=False, bias=1.0),
                         validation_steps = (len(val_data)/CONFIG['batchsize']+1),
                         callbacks=[checkpointer, logger])
