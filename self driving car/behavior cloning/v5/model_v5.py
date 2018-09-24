from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense, Dropout, Lambda, LeakyReLU
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

    input_frame_x = Input(shape=(NVIDIA_H, NVIDIA_W, CONFIG['input_channels']))

    # standardize input
    input_x = Lambda(lambda z: z / 127.5 - 1.)(input_frame_x)

    x = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), init=init, activation='relu')(input_x)
    x = Dropout(0.2)(x)
    x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), init=init, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), init=init,activation='relu' )(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', init=init,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', init=init,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    x = Dense(100, init=init)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(50, init=init)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(10, init=init)(x)
    x = LeakyReLU()(x)
    steer_out = Dense(1, init=init, name="steer")(x)
    speed_out = Dense(1, init=init, name="speed")(x)
    
    model = Model(input=input_frame_x, output=[steer_out,speed_out])

    if summary:
        model.summary()
    return model


if __name__ == '__main__':

    # split udacity csv data into training and validation
    train_data, val_data = split_train_val(csv_driving_data='data/driving_log-raymond-and-narrow-prev-frame.csv')

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
    nvidia_net.compile(optimizer='adam', loss=['mae','mae'], loss_weights=[1.0,1.0])

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
