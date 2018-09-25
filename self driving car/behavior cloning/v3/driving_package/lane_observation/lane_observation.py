import numpy as np
from collections import Counter

import cv2
from sklearn.neighbors import KDTree, BallTree


class LaneObserver(object):
    """
        Help to observe the state of lane
    """

    LANE_TYPE_WIDE = 0
    LANE_TYPE_NARROW = 1

    GREEN = 0
    RED = 1
    BLUE = 2
    BLACK = 3
    YELLOW = 4
    WHITE = 5

    # LAB config
    LAB_GREEN_TRACK = (168, 89, 163)
    LAB_RED_TRACK = (150, 185, 163)
    LAB_BLUE_TRACK = (132, 126, 106)
    LAB_BLACK_WALL = (2, 128, 128)
    LAB_YELLOW_WALL = (172, 113, 197)
    LAB_WHITE_SKY = (255, 128, 128)

    def __init__(self):
        self._color_list = [
            self.LAB_GREEN_TRACK,  # 0
            self.LAB_RED_TRACK,   # 1
            self.LAB_BLUE_TRACK,  # 2
            self.LAB_BLACK_WALL,  # 3
            self.LAB_YELLOW_WALL,  # 4
            self.LAB_WHITE_SKY    # 5
        ]

        assert self._color_list[self.GREEN] == self.LAB_GREEN_TRACK
        assert self._color_list[self.RED] == self.LAB_RED_TRACK
        assert self._color_list[self.BLUE] == self.LAB_BLUE_TRACK
        assert self._color_list[self.BLACK] == self.LAB_BLACK_WALL
        assert self._color_list[self.YELLOW] == self.LAB_YELLOW_WALL
        assert self._color_list[self.WHITE] == self.LAB_WHITE_SKY

        self._color_str = {
            self.GREEN: 'g',
            self.RED: 'r',
            self.BLUE: 'b',
            self.BLACK: 'B',
            self.YELLOW: 'Y',
            self.WHITE: 'W'
        }

        self._color_kd_tree = BallTree(self._color_list)
        self._prev_lane_type = LaneObserver.LANE_TYPE_WIDE

    def predict(self, rgb_img, curr_speed, curr_steering_angle):
        """
        Parameters
            rgb_img: (numpy.ndarray) rgb-channels image
            curr_speed: current speed
            curr_steering_angle: current steering angle
        Returns:
            LaneObserver.LANE_TYPE_WIDE = 0
            LaneObserver.LANE_TYPE_NARROW = 1
        """
        copy_img = np.copy(rgb_img)
        copy_img[0:120, :] = (255, 255, 255)

        lab = cv2.cvtColor(copy_img, cv2.COLOR_RGB2LAB)
        flat_lab = lab.reshape(-1, 3)

        # Find Matched color
        min_idx = self._color_kd_tree.query(flat_lab, return_distance=False)
        min_idx = min_idx.reshape(lab.shape[0:2])

        lab_new_image = np.zeros_like(lab)
        for i, color in enumerate(self._color_list):
            lab_new_image[min_idx == i] = color

        bird_view_lab = BirdViewTransform.process(lab_new_image, border_value=LaneObserver.LAB_WHITE_SKY)
        bird_view_lab = self.__exclued_small_yellow(bird_view_lab)

        ref_line_y = 65
        ref_line = bird_view_lab[ref_line_y:ref_line_y+1, :]
        ref_line = ref_line.reshape(-1, 3)

        ref_line_idx = self._color_kd_tree.query(ref_line, return_distance=False)

        ref_line_idx = ref_line_idx[ref_line_idx != self.WHITE]

        # Apply mode filter witk kernel size = 23
        ref_line_idx = self.__mode_filter(ref_line_idx, 23)

        lane_type = self.__determine_lane_type(ref_line_idx)

        # if lane_type == LaneObserver.LANE_TYPE_NARROW:
        #     cv2.putText(rgb_img, 'Narrow', (20, 20), cv2.FONT_ITALIC, 0.4, (255, 0, 0), 1)
        # else:
        #     cv2.putText(rgb_img, 'Wide', (20, 20), cv2.FONT_ITALIC, 0.4, (0, 0, 255), 1)
        # show_image(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), 'raw_2', 1)
        # show_image(cv2.cvtColor(bird_view_lab, cv2.COLOR_LAB2BGR), 'bird_view', 1)
        # if lane_type == LaneObserver.LANE_TYPE_NARROW:
        #     cv2.waitKey(0)

        return lane_type

    def __determine_lane_type(self, color_seq):
        if len(color_seq) < 3:
            return self._prev_lane_type

        color_seq_str = ''
        curr_color = -1
        for color in color_seq:
            if curr_color == color:
                continue
            color_seq_str += self._color_str[color]
            curr_color = color

        # Update lane type
        if 'rg' in color_seq_str:
            self._prev_lane_type = LaneObserver.LANE_TYPE_WIDE
        elif 'Y' in color_seq_str:
            self._prev_lane_type = LaneObserver.LANE_TYPE_NARROW

        return self._prev_lane_type

    def __mode_filter(self, seq, kernel_size=3):
        if len(seq) < kernel_size:
            return seq

        half = int((kernel_size - 1) / 2)
        for i in range(half, len(seq) - half):
            sub_seq = seq[i-half: i + half + 1]
            counter = Counter(sub_seq)
            seq[i] = counter.most_common(1)[0][0]

        return seq

    def __exclued_small_yellow(self, lab_image):
        yellow_blob = self.__find_blob_with_lab(lab_image, self.LAB_YELLOW_WALL)
        n_labels, labels, ccl_status, _ = cv2.connectedComponentsWithStats(yellow_blob)

        for i, status in zip(range(n_labels), ccl_status):
            if i == 0:
                continue
            if status[4] < 2000:  # area
                lab_image[labels == i] = self.LAB_WHITE_SKY
        return lab_image

    def __find_blob_with_lab(self, lab_image, blob_lab_range):
        L_RANGE = 10
        lower_b = tuple([x - L_RANGE for x in blob_lab_range])
        upper_b = tuple([x + L_RANGE for x in blob_lab_range])
        return cv2.inRange(lab_image, lower_b, upper_b)


class BirdViewTransform:
    __trans_matrix = [
        [-9.79099981e-02, -1.49124459e+00, 1.95616645e+02],
        [-4.99600361e-15, -2.98499969e+00, 3.86445742e+02],
        [-1.32272665e-17, -8.28469215e-03, 1.00000000e+00]]

    @staticmethod
    def process(src, border_value=(255, 255, 255)):
        return cv2.warpPerspective(src, np.array(BirdViewTransform.__trans_matrix, np.float32), (360, 360),
                                   borderValue=border_value, flags=cv2.INTER_NEAREST)


def show_image(img, name='', wait_time=0):
    cv2.imshow(name, img)
    if cv2.waitKey(wait_time) == 27:
        exit(0)


if __name__ == '__main__':

    import os
    import glob
    import json

    import base64
    from PIL import Image
    from io import BytesIO

    # record_file = '../driving-logs/20180918-full-speed/Track1/2018-09-18T10-13-49-lap02.json'
    # record_file = '../driving-logs/20180918-full-speed/Track2/2018-09-18T10-37-26-lap01.json'
    # record_file = '../driving-logs/2018-09-06T17-37-27-lap01.json'
    # record_file = '../driving-logs/Track2/2018-09-05T20-57-31-lap01.json'
    # record_file = '../driving-logs/Track5/2018-09-06T10-43-36-lap01.json'
    # record_file = '../driving-logs/Track5/2018-09-03T16-08-52-lap01.json'
    record_file = 'driving-logs/2018-09-19T22-39-56-lap03.json'
    lane_pos = LaneObserver()
    with open(record_file) as f:
        record_json = json.load(f)

        for record in record_json['records']:
            img = np.asarray(Image.open(BytesIO(base64.b64decode(record["image"]))))
            curr_speed = record['curr_speed']
            curr_steering_angle = record['curr_steering_angle']
            cmd_speed = record['cmd_speed']
            cmd_steering_angle = record['cmd_steering_angle']
            lane_pos.predict(img, curr_speed, curr_steering_angle)
