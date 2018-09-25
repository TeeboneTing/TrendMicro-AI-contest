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
from config_v3 import *
from load_data_v3 import preprocess
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import sys

from driving_package.sign_detector import SignDetector
from driving_package.lane_observation import LaneObserver

# V3.1 using PID control to get throttle value
from sample_bot import PID

THROTTLE_PID_Kp             = 0.26
THROTTLE_PID_Ki             = 0.0
THROTTLE_PID_Kd             = 0.007
THROTTLE_PID_max_integral   = 1.0
throttle_pid                = PID(Kp=THROTTLE_PID_Kp  , Ki=THROTTLE_PID_Ki  , Kd=THROTTLE_PID_Kd  , max_integral=THROTTLE_PID_max_integral)

sio = socketio.Server()
app = Flask(__name__)

#backward_switch = False
#backward_counter = 0

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

    sign_array, sign_name = sign_det.detect(np.asarray(image))
    print(sign_name)

    lane_type = lane_det.predict(np.asarray(image), data['speed'], data['steering_angle'])

    #if sign_name != "Nothing":
    #    print("Sing Detected: {sign_name}")

    # frames incoming from the simulator are in RGB format
    image_array = cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR)

    # perform preprocessing (crop, resize etc.)
    image_array = preprocess(frame_bgr=image_array)

    # add singleton batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    
    # Turn off backward mode in Beta test
    """
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
    """
    # V3.1: prediction including steering and cmd_speed.
    # Using PID control to get throttle from cmd_speed and current speed

    if lane_type == LaneObserver.LANE_TYPE_NARROW:
        print('Narrow')
        steering_angle, cmd_speed = narrow_model.predict(image_array, batch_size=1)
        cmd_speed = 1.3
    else:
        steering_angle, cmd_speed = model.predict(image_array, batch_size=1)
        cmd_speed -= 0.1
    steering_angle = float(steering_angle)

    if sign_name == "ForkLeft":
        steering_angle += -3
        cmd_speed = 1.3
    elif sign_name == "ForkRight":
        steering_angle += 3
        cmd_speed = 1.3

    cmd_speed = float(cmd_speed)
    #throttle = throttle_control(0.02,speed,steering_angle)

    throttle_pid.assign_set_point(cmd_speed)
    throttle = throttle_pid.update(speed)
    
    """
    if backward_switch == True:
        steering_angle = -40 #-steering_angle*5
        throttle = -throttle*5
        backward_counter += 1
        if backward_counter > 15*6:
            backward_switch = False
            backward_counter = 0
    """


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
    json_path = 'model_weights_v3.1/model-wide.json'
    with open(json_path) as jfile:
        model = model_from_json(jfile.read())

    # load model weights
    weights_path = 'model_weights_v3.1/model-wide-weights.68-3.679.hdf5'
    print('Loading weights: {}'.format(weights_path))
    model.load_weights(weights_path)

    # load narrow lane model from json
    json_path = 'model_weights_v3.1/model-narrow.json'
    with open(json_path) as jfile:
        narrow_model = model_from_json(jfile.read())

    # load narrow lane model weights
    weights_path = 'model_weights_v3.1/model-narrow-weights.33-4.037.hdf5'
    print('Loading weights: {}'.format(weights_path))
    narrow_model.load_weights(weights_path)

    MODEL_SIGN = "v3/model/sign_16x32_3232_8_201809152226_0.9993_0.9703_53k.hdf5"
    sign_det = SignDetector(MODEL_SIGN)
    lane_det = LaneObserver()

    

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
