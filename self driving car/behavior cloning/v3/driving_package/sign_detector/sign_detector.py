# -*- coding: utf-8 -*-
import cv2
import numpy as np
from keras.models import load_model

import pdb

#from driving_package import ers_bot_logger
#logger = ers_bot_logger.get_logger(__name__)


class SignDetector:
    LAB_SIGN = (100, 175, 170)
    SIGN_MAP = {0: 'ForkLeft',    1: 'ForkRight',
                2: 'TurnLeft',    3: 'TurnRight',
                4: 'UTurnLeft',   5: 'UTurnRight',
                6: 'WarningLeft', 7: 'WarningRight', 8: 'Nothing'}

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def __find_blob_with_lab(self, lab_image, blob_lab_range, L_RANGE=30):
        lower_b = tuple([x - L_RANGE for x in blob_lab_range])
        upper_b = tuple([x + L_RANGE for x in blob_lab_range])
        return cv2.inRange(lab_image, lower_b, upper_b)

    def __find_sign_bbox(self, lab_image, shift=(0, 0), padding=1):
        # find
        mask = self.__find_blob_with_lab(lab_image, self.LAB_SIGN)
        if np.max(mask) != 255:
            return None, None
        # roi
        ax, ay = np.where(mask == 255)
        top    = max(min(ax) - padding + shift[0], 0) 
        left   = max(min(ay) - padding + shift[1], 0) 
        # add 1 for indexing
        bottom = min(max(ax) + 1 + padding + shift[0], lab_image.shape[0] + shift[0])
        right  = min(max(ay) + 1 + padding + shift[1], lab_image.shape[1] + shift[1])
        return (top, left, bottom, right), mask

    def bbox2area(self, bbox):
        top, left, bottom, right = bbox
        area = (bottom - top) * (right - left)
        return area

    def crop_sign(self, lab_image, bbox_thres=60):
        # process
        img_sign = None
        bb, mask = self.__find_sign_bbox(lab_image)
        if not bb:
            return None, None
        # reindex due to two sign
        top, left, bottom, right = bb
        if right - left > bbox_thres:
            idx_mid = int((right + left)/2)
            bb1, _ = self.__find_sign_bbox(lab_image[:, :idx_mid])
            bb2, _ = self.__find_sign_bbox(lab_image[:, idx_mid:], shift=(0, idx_mid))
            if self.bbox2area(bb1) > self.bbox2area(bb2):
                bb = bb1
            else:
                bb = bb2
            top, left, bottom, right = bb
        # crop
        img_sign = lab_image[top:bottom, left:right, :]
        mask = mask[top:bottom, left:right]
        return img_sign, mask

    def detect(self, rgb_image):
        # init
        label_num = len(self.SIGN_MAP) - 1
        # process image
        #pdb.set_trace()
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        sign_img, mask = self.crop_sign(lab_image[:100])
        # process detect
        output = np.array([0]*label_num + [1])
        if isinstance(sign_img, np.ndarray):
            h, w, _ = sign_img.shape
            area = h * w
            roi_hot = np.count_nonzero(mask) / area
            r_shape = h / w
            if area > 121 and r_shape > 0.2 and r_shape < 0.8 and roi_hot > 0.1:
                sign_img = cv2.cvtColor(sign_img, cv2.COLOR_LAB2RGB)
                sign_img = cv2.resize(sign_img, (32, 16), interpolation=cv2.INTER_CUBIC)
                sign_img = cv2.cvtColor(sign_img, cv2.COLOR_RGB2GRAY)
                input_img = np.expand_dims(sign_img, 2)
                input_img = np.expand_dims(input_img, 0)
                output = self.model.predict(input_img)
                output = np.concatenate([output[0], [0]])
        sign_idx  = int(np.argmax(output))
        sign_name = self.SIGN_MAP[sign_idx]
        return output, sign_name

    def move_process(self, sign_name):
        pass


if __name__ == "__main__":
    """
    Setting:
        LAB_SIGN = (100, 175, 170), L_RANGE=30
        crop_sign: bbox_thres=60
        if area > 121 and r_shape > 0.2 and r_shape < 0.8 and roi_hot > 0.1:

                  Hit  false         Nothing
    ForkLeft:     163      3  0.982  3
    ForkRight:    163      3  0.982  3
    TurnLeft:      78      1  0.987  1
    TurnRight:     77      2  0.975  2
    UTurnLeft:     56      0  1.000  0
    UTurnRight:    56      0  1.000  0
    """
    import glob
    import os
    from PIL import Image

    model_path = r"../../model/sign_16x32_3232_8_201809152226_0.9993_0.9703_53k.hdf5"
    model = SignDetector(model_path)

    # Traffic Sigh Detection - Benchmark
    test_data = r"D:\Trend\Al_game_2018_global\League\League-Traffic-Sign"
    test_data = glob.glob(os.path.join(test_data, "*", "*.jpg"))

    result = {}
    for fpath in test_data:
        y = os.path.basename(os.path.dirname(fpath))
        y = y.replace(" ", "").replace("-", "")
        rgb_img = np.array(Image.open(fpath))
        _, pred = model.detect(rgb_img)
        # calculate
        if y not in result:
            result[y] = {"Hit": 0, "false alarm": 0,
                         "fa_detail": [], "Nothing": []}
        if y == pred:
            result[y]["Hit"] += 1
            continue
        result[y]["false alarm"] += 1
        if pred == "Nothing":
            result[y]["Nothing"].append(fpath)
        else:
            result[y]["fa_detail"].append((pred, fpath, rgb_img))

    for k, v in result.items():
        print('{k}: {v["Hit"]/(v["Hit"]+v["false alarm"])}')
