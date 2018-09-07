import os
import glob
import json
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import argparse
import glob

import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Game Log player')
    parser.add_argument('--img-folder', metavar='FOLDER',
                        type=str, nargs='?',
                        help='IMG folder to play')

    args = parser.parse_args()

    if not any([args.img_folder]):
        parser.print_help()
        exit(0)

    record_file_to_play = []
    if args.img_folder:
        print(args.img_folder)
        files = glob.glob(os.path.join(args.img_folder, '*.jpg'))
        files = sorted(files)
        record_file_to_play.extend(files)
    print ('Play record folder: %s' % args.img_folder)
    for record_img in record_file_to_play:
        img = cv2.cvtColor(np.asarray(Image.open(record_img)), cv2.COLOR_RGB2BGR)

        cv2.imshow('image', img)
        if cv2.waitKey(50) == 27:  # ESC to exit
            exit(0)
