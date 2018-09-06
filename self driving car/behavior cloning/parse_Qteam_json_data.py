# parse_Qteam_json_data.py
# Goal: transfer Qteam's json data into Udacity driving_log.csv + IMG/ format
# usage: parse_Qteam_json_data.py [-h] [--json-folder FOLDER 
#                                 [--output-folder FOLDER ]

import os
import glob
import json
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import argparse
import cv2
import pdb
import csv

from sample_bot import PID

"""
json format:
{
    "lap": lap index
    "elapsed_time": elapsed time during this lap
    "records": [
        {
            "image": base64 image
            "curr_speed": current speed from dashboard. unit: m/s
            "curr_steering_angle": current steering angle from dashboard. unit: degree
            "cmd_speed": speed command from human player. unit: m/s
            "cmd_steering_angle": steering angle command from human player. unit: degree
            "time": timestamp. unit: seconds
        } 
        ... list of record
    ]
}

csv schema:
ct_path, steer, throttle, brake, speed, time_stamp, lap
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Goal: transfer Qteam's json data into Udacity driving_log.csv + IMG/ format.")
    parser.add_argument('--json-folder', metavar='FOLDER',
                        type=str, nargs='?',
                        help='Json record folder as Input')
    parser.add_argument('--output-folder', metavar='FOLDER',
                        type=str, nargs='?',
                        help='Output file folder')

    args = parser.parse_args()
    return vars(args)



def run():
    arguments = parse_args()
    if arguments['output_folder'] == None or arguments['json_folder'] == None:
        print("Please provide json file folder and output folder names.")
        exit(1)
    #print(arguments)

    # Read json files into memory
    jfile_list = glob.glob(arguments['json_folder']+"/*")
    f_prefix = [ jfile_name.split("\\")[-1].split('.')[0] for jfile_name in jfile_list]
    #print(f_prefix)
    json_data = []
    for jfile in jfile_list:
        with open(jfile) as f:
            json_data.append(json.load(f))
    
    # Create folder and IMG subfolder 
    if not os.path.exists(arguments["output_folder"]+"/IMG"):
        os.makedirs(arguments["output_folder"]+"/IMG")

    # In order to calculate throttle, using PID control to get throttle values
    THROTTLE_PID_Kp             = 0.02
    THROTTLE_PID_Ki             = 0.005
    THROTTLE_PID_Kd             = 0.02
    THROTTLE_PID_max_integral   = 0.5
    throttle_pid                = PID(Kp=THROTTLE_PID_Kp  , Ki=THROTTLE_PID_Ki  , Kd=THROTTLE_PID_Kd  , max_integral=THROTTLE_PID_max_integral)
    #self._throttle_pid.assign_set_point(self.DEFAULT_SPEED)

    for f_idx,one_file_data in enumerate(json_data):
        print("Exporting data from json file: %s"%jfile_list[f_idx])
        driving_log = []
        for idx,record in enumerate(one_file_data["records"]):
            try:
                # Read imgs
                img = Image.open(BytesIO(base64.b64decode(record["image"])))    
                
                # Save image to output folder with sub-folder IMG
                ct_path = arguments["output_folder"]+"/IMG/%s-%d"%(f_prefix[f_idx],idx)+".jpg"
                img.save(ct_path,"JPEG")
                
                # Read driving log
                steer         = record["curr_steering_angle"]
                throttle_pid.assign_set_point(record["cmd_speed"])
                throttle      = throttle_pid.update(record["curr_speed"])
                brake         = 0.0
                speed         = record["curr_speed"]
                time_stamp    = record["time"]
                lap           = one_file_data["lap"]
            except:
                pdb.set_trace()
            # Append data
            driving_log.append([ct_path,steer,throttle,brake,speed,time_stamp,lap])
        #pdb.set_trace()
           
        # Save driving_log
        with open(arguments["output_folder"] + '/driving_log-%s.csv'%(f_prefix[f_idx]), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for data in driving_log:
                writer.writerow(data)

if __name__ == "__main__":
    run()