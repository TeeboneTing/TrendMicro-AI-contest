# Bump Detector

Speed based + Wall Color based bumping detector, and detect which side of wall bumped.

## Note

Currently, we can not detect bumping into obstacle, but coming soon...



## Demo

![Demo](doc/demo.gif)

## Pre-requirement
```
opencv-python >= 3.0.0
```


## Usage
It's necessary to call BumpDetector.detect realtime.

### Sample Code
```python
import numpy as np
import base64
from PIL import Image
from io import BytesIO

from bump_detector import BumpDetector

bump_detector = BumpDetector()

@sio.on('telemetry')
def telemetry(sid, dashboard):
    curr_speed = float(dashboard['speed']) 
    curr_throttle = float(dashboard['throttle'])
    rgb_image = np.asarray(Image.open(BytesIO(base64.b64decode(data['image']))

    bump_status = bump_detector.detect(rgb_image, curr_speed, curr_throttle)
    if bump_status == BumpDetector.NO_BUMP:
        pass
    elif bump_status == BumpDetector.BUMP_INTO_LEFT_WALL:
        print ('Bump Left Wall!!!')
        # Try to steer left and reverse.
    elif bump_status == BumpDetector.BUMP_INTO_RIGHT_WALL:
        print ('Bump Left Wall!!!')
        # Try to steer right and reverse.
    else:
        print ('Bump anything!!!')
        # Try to reverse ,maybe...
        
    
    ...
```
## API

### BumpDetector

### detect(rgb_image, curr_speed, curr_throttle)

+ Parameters:
    + rgb_img: numpy.ndarray with rgb channels
    + curr_speed: current speed from dashboard (m/s)
    + curr_throttle: current throttle from dashboard
+ Retuens:
    + NO_BUMP = 0
    + BUMP_INTO_LEFT_WALL = 1
    + BUMP_INTO_RIGHT_WALL = 2
    + BUMP_ANYTHING = 99