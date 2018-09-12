# Add traffic sign label by JPG file names and driving log

import argparse
import glob
import pdb
import pandas as pd
import numpy as np


"""
Current driving log schema:
    img_path, steer, throttle, brake, speed, time_stamp, lap

New driving log schema:
    img_path, steer, throttle, brake, speed, time_stamp, lap, have_traffic_sign, sign_category

Sign categories:
    No traffic sign:  0
    ForkLeft:         1 # important 
    ForkRight:        2 # important 
    WarningLeft:      3 # important 
    WarningRight:     4 # important 
    TurnLeft:         5 # optional
    TurnRight:        6 # optional
    UTurnLeft:        7 # optional
    UTurnRight:       8 # optional
    
"""

def run():
    args = parse_args()
    #print(args)
    traffic_sign_dict = read_traffic_signs(args["label_folder"])
    original_driving_log = pd.read_csv(args["driving_log"],header=None)
    new_driving_log = add_traffic_sign_features(original_driving_log,traffic_sign_dict)
    new_driving_log.to_csv(args["output"],header=False,index=False)

def add_traffic_sign_features(driving_log,traffic_sign_dict):
    have_traffic_sign = np.zeros(driving_log.shape[0],dtype=np.int)
    sign_category     = np.zeros(driving_log.shape[0],dtype=np.int)
    sign2cat = {
        "ForkLeft":     1,
        "ForkRight":    2,
        "WarningLeft":  3,
        "WarningRight": 4,
        "TurnLeft":     5,
        "TurnRight":    6,
        "UTurnLeft":    7,
        "UTurnRight":   8}
    for idx, img_path in enumerate(driving_log[0]):
        img_filename = img_path.split("/")[-1]
        for sign, file_list in traffic_sign_dict.items():
            if img_filename in file_list:
                have_traffic_sign[idx] = 1
                sign_category[idx] = sign2cat[sign]
    original_columns = driving_log.shape[1]
    driving_log[original_columns] =   have_traffic_sign
    driving_log[original_columns+1] = sign_category
    return driving_log


def read_traffic_signs(label_folder,only_important=True):
    traffic_signs = {
       "ForkLeft":     [],
       "ForkRight":    [],
       "WarningLeft":  [],
       "WarningRight": []}
    if only_important == False: # add all traffic signs
        traffic_signs["TurnLeft"]  = []
        traffic_signs["TurnRight"] = []
        traffic_signs["UTurnLeft"] = []
        traffic_signs["UTurnRight"]= []
    file_paths = glob.glob(label_folder + "/*/*.jpg")
    #print(file_paths[:10])
    for path in file_paths:
        split = path.split("/")
        img_filename, sign = split[-1], split[-2]
        if sign in traffic_signs.keys():
            traffic_signs[sign].append(img_filename)
    return traffic_signs

def parse_args():
    parser = argparse.ArgumentParser(description='Add traffic sign label to driving log')
    parser.add_argument('--label-folder', metavar='FOLDER',
                        type=str, nargs='?',
                        help='label folder for traffic sign pictures')
    parser.add_argument('--driving-log', metavar='LOGFILE',
                        type=str, nargs='?',
                        help='driving log file')
    parser.add_argument('--output', metavar='NEWLOGFILE',
                        type=str, nargs='?',
                        help='new driving log file')

    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    run()
