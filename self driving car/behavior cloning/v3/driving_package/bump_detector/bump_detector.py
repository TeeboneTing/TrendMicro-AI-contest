import numpy as np
import time
from collections import deque

import cv2


class BumpDetector:
    NO_BUMP = 0
    BUMP_INTO_LEFT_WALL = 1
    BUMP_INTO_RIGHT_WALL = 2
    BUMP_ANYTHING = 99

    IMG_QUEUQ_SIZE = 5

    def __init__(self):
        self._speed_detector = SpeedBumpDetector()
        self._color_wall_detector = ColorWallBumpDetector()
        self._rgb_img_queue = deque(maxlen=BumpDetector.IMG_QUEUQ_SIZE)

        self._prev_bump_status = BumpDetector.NO_BUMP
        self._prev_bump_time = 0

    def detect(self, rgb_image, curr_speed, curr_throttle):
        """
        Args:
            rgb_img: np.ndarray with rgb channels
        Retuens:
            bump_status:
                NO_BUMP = 0
                BUMP_INTO_LEFT_WALL = 1
                BUMP_INTO_RIGHT_WALL = 2
                BUMP_ANYTHING = 99
        """
        self._rgb_img_queue.append(rgb_image)
        is_bump = self._speed_detector.detect(curr_speed, curr_throttle) or self._color_wall_detector.detect(rgb_image)

        if not is_bump:
            self._prev_bump_status = BumpDetector.NO_BUMP
            return BumpDetector.NO_BUMP

        # A workaround to prevent bump status change rapidly
        if time.time() - self._prev_bump_time < 1.0 and self._prev_bump_status != BumpDetector.NO_BUMP:
            self._prev_bump_time = time.time()
            return self._prev_bump_status

        self._prev_bump_time = time.time()
        self._prev_bump_status = self._color_wall_detector.detect_bump_side(list(self._rgb_img_queue))
        return self._prev_bump_status


class SpeedBumpDetector:
    def __init__(self):
        self._prev_speed = 0
        self._bump_timestamp = deque(maxlen=1000)

    def detect(self, curr_speed, throttle):
        is_bump = False
        throttle = abs(throttle)
        if self._prev_speed != 0:
            if (curr_speed - self._prev_speed) < -0.5:
                is_bump = True

        if throttle > 0.1 and abs(curr_speed) < 0.05:
            self._bump_timestamp.append(time.time())
        else:
            self._bump_timestamp.clear()
        try:
            time_diff = self._bump_timestamp[-1] - self._bump_timestamp[0]
        except IndexError:
            pass
        else:
            if time_diff > 1.0:
                is_bump = True

        self._prev_speed = curr_speed
        return is_bump


class ColorWallBumpDetector:
    LAB_BLACK_WALL = (2, 128, 128)
    LAB_YELLOW_WALL = (172, 113, 197)

    def __init__(self):
        pass

    def detect(self, rbg_img):
        """
        Args:
            rgb_img: np.ndarray with rgb channels
        Retuens:
            bool: True if bump into the wall
        """

        lab_image = cv2.cvtColor(rbg_img, cv2.COLOR_RGB2LAB)
        wall_blob = self.__find_wall_blob(lab_image)
        n_labels, _, ccl_status, _ = cv2.connectedComponentsWithStats(wall_blob)
        if n_labels < 2:
            return False

        wall_area_list = ccl_status[1:, 4]
        max_area = max(wall_area_list)
        ratio = max_area / lab_image.shape[0] / lab_image.shape[1]
        return ratio >= 0.75

    def __find_wall_blob(self, lab_image):
        black_wall = self.__find_blob_with_lab(lab_image, self.LAB_BLACK_WALL)
        yellow_wall = self.__find_blob_with_lab(lab_image, self.LAB_YELLOW_WALL)
        all_wall = black_wall + yellow_wall
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.morphologyEx(all_wall, cv2.MORPH_CLOSE, kernel)

    def __find_blob_with_lab(self, lab_image, blob_lab_range):
        L_RANGE = 10
        lower_b = tuple([x - L_RANGE for x in blob_lab_range])
        upper_b = tuple([x + L_RANGE for x in blob_lab_range])
        return cv2.inRange(lab_image, lower_b, upper_b)

    def detect_bump_side(self, rgb_image_seqs):
        score_left = 0
        score_right = 0

        for rgb_img in rgb_image_seqs:
            lab_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
            wall_blob = self.__find_wall_blob(lab_image)
            wall_blob_bird_view = BirdViewTransform.process(wall_blob, (0))

            half_width = int(wall_blob_bird_view.shape[1] / 2)
            left = wall_blob_bird_view[:, 0:half_width]
            right = wall_blob_bird_view[:, half_width:]
            left_wall_ratio = np.sum(left == 255) / left.size
            right_wall_ratio = np.sum(right == 255) / right.size

            if abs(left_wall_ratio - right_wall_ratio) > 0.005:
                if left_wall_ratio > right_wall_ratio:
                    score_left += left_wall_ratio

                else:
                    score_right += right_wall_ratio

        if score_left > score_right:
            return BumpDetector.BUMP_INTO_LEFT_WALL
        elif score_left < score_right:
            return BumpDetector.BUMP_INTO_RIGHT_WALL
        else:
            return BumpDetector.BUMP_ANYTHING  # TODO: How to detect bump into obstacle


class BirdViewTransform:
    __trans_matrix = [
        [-9.79099981e-02, -1.49124459e+00, 1.95616645e+02],
        [-4.99600361e-15, -2.98499969e+00, 3.86445742e+02],
        [-1.32272665e-17, -8.28469215e-03, 1.00000000e+00]]

    @staticmethod
    def process(src, border_value=(255, 255, 255)):
        return cv2.warpPerspective(src, np.array(BirdViewTransform.__trans_matrix, np.float32), (360, 360),
                                   borderValue=border_value, flags=cv2.INTER_LINEAR)
