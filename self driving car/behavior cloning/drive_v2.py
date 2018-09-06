#import argparse
import base64
import json
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import os
import numpy as np
from config import *
from load_data_v2 import preprocess
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import sys

# Fix error with Keras and TensorFlow
import tensorflow as tf

#tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
backward_switch = False
backward_counter = 0

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))

    time = float(data["time"])

    lap = int(data["lap"])

    brakes = float(data["brakes"])    

    # frames incoming from the simulator are in RGB format
    image_array = cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR)

    # perform preprocessing (crop, resize etc.)
    image_array = preprocess(frame_bgr=image_array)

    # add singleton batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    
    global backward_switch
    global backward_counter
    if speed < 0.01:
        backward_counter += 1
        print(backward_counter)
        if backward_counter > 15*2:
            backward_switch = True
            backward_counter = 0
    if speed > 0.1 and backward_switch == False:
        backward_counter = 0

    steering_angle = float(model.predict(image_array, batch_size=1))
    throttle = throttle_control(0.02,speed,steering_angle)
    if backward_switch == True:
        steering_angle = -40 #-steering_angle*5
        throttle = -throttle*5
        backward_counter += 1
        if backward_counter > 15*6:
            backward_switch = False
            backward_counter = 0


    print(speed,steering_angle, throttle)
    send_control(steering_angle, throttle)

def throttle_control(default_throttle,current_speed,steering_angle):
    if abs(steering_angle) > 1.0 and abs(steering_angle) <= 5.0 and current_speed > 1.0:
        throttle = 0.01
    elif abs(steering_angle) > 5.0:
        throttle = 0.005
    else:
        throttle = default_throttle
    return throttle

def backward_control():
    global backward_switch
    global backward_counter
    if backward_counter < 15*100:
        backward_counter += 1
    else:
        backward_switch = False
        backward_counter = 0
    throttle = -1.0
    steering_angle = 40.0
    return throttle,steering_angle

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':

    from keras.models import model_from_json

    # load model from json
    #json_path ='pretrained/model.json'
    #json_path ='logs/model.json'
    json_path = sys.argv[1]
    with open(json_path) as jfile:
        model = model_from_json(jfile.read())

    # load model weights
    # weights_path = os.path.join('checkpoints', os.listdir('checkpoints')[-1])
    #weights_path = 'pretrained/model.hdf5'
    weights_path = sys.argv[2]
    print('Loading weights: {}'.format(weights_path))
    model.load_weights(weights_path)

    # compile the model
    #model.compile("adam", "mse")
    model.compile("sgd", "mse")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
