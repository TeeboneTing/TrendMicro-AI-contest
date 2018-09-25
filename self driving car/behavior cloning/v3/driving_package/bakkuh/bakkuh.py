# -*- coding: utf-8 -*-
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO

from driving_package import ers_bot_logger
logger = ers_bot_logger.get_logger(__name__)

class Bakkuh:
    BAKKUH_DIRECTION_LEFT = 0
    BAKKUH_DIRECTION_RIGHT = 1
    BAKKUH_DIRECTION_STRIGHT = 2 # TODO: implement

    BAKKUH_STATUS_DONE = 0
    BAKKUH_STATUS_RUNNING = 1

    def __find_blob_with_lab(self, lab_image, blob_lab_range):
        L_RANGE = 10
        lower_b = tuple([x - L_RANGE for x in blob_lab_range])
        upper_b = tuple([x + L_RANGE for x in blob_lab_range])
        return cv2.inRange(lab_image, lower_b, upper_b)

    def __find_color_ratio(self, rgb_image, blob_lab_range, bbox):
        top, left, bottom, right = bbox
        ref_img = rgb_image[top:bottom, left:right]
        color_blob = self.__find_blob_with_lab(cv2.cvtColor(ref_img, cv2.COLOR_RGB2LAB), blob_lab_range)
        color_area = np.count_nonzero(color_blob)
        bbox_area = ref_img.shape[0] * ref_img.shape[1]
        color_ratio = color_area / float(bbox_area)
        return color_ratio

    def process(self, rgb_image, bakkuh_direction,
                ref_bbox=(239, 0, 241, 320), thres=0.8):
        """
        Returns: tuple
            (BAKKUH_STATUS, throttle, steering_angle)
        """
        is_left = bakkuh_direction == Bakkuh.BAKKUH_DIRECTION_LEFT
        LAB_BLUE_TRACK = (132, 126, 106)

        # check if baku to track
        ref_bbox_side = (100, 319, 241, 321) if is_left else(100, 0, 241, 2)
        blue_ratio_main = self.__find_color_ratio(rgb_image, LAB_BLUE_TRACK, ref_bbox)
        blue_ratio_side = self.__find_color_ratio(rgb_image, LAB_BLUE_TRACK, ref_bbox_side)
        blue_ratio = max(blue_ratio_main, blue_ratio_side)

        if blue_ratio > thres:
            return Bakkuh.BAKKUH_STATUS_DONE, 0, 0 # Bakkuh done
        else:  # action
            steer_base = -35 if is_left else 35
            steer_ratio = (thres - blue_ratio) / thres * 2
            throttle = -0.125 * steer_ratio
            return  Bakkuh.BAKKUH_STATUS_RUNNING, throttle, steer_base * steer_ratio


if __name__ == "__main__":
    from driving_package.bump_detector import BumpDetector

    bump_detector = BumpDetector()
    bakkuh = Bakkuh()

    @sio.on('telemetry')
    def telemetry(sid, dashboard):
        curr_speed = float(dashboard['speed'])
        curr_throttle = float(dashboard['throttle'])
        rgb_image = np.asarray(Image.open(BytesIO(base64.b64decode(dashboard['image']))))
        # in realtime
        bump_status = bump_detector.detect(rgb_image, curr_speed, curr_throttle)
        if bump_status:
            steer, throttle = bakkuh.process(rgb_image, bump_status)
            return steer, throttle
