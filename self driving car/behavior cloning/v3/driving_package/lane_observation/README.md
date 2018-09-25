```python

class LaneObserver(object):
    LANE_TYPE_WIDE = 0
    LANE_TYPE_NARROW = 1

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
```