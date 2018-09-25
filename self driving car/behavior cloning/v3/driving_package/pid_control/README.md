# PID Controller

Control everything what you want with PID (proportional–integral–derivative)

## Speed PID Control

### Usage

An well-performed Speed PID parameters I have tuned.

```python
from pid_control import PIDController

SPEED_PID_KP = 0.26
SPEED_PID_KI = 0.015
SPEED_PID_KD = 0.007
MAX_THROTTLE = 1

speed_pid = PIDController(
    Kp=SPEED_PID_KP, Ki=SPEED_PID_KI, Kd=SPEED_PID_KD,
    max_I=MAX_THROTTLE, max_output=MAX_THROTTLE)

def on_telemetry(data):

    # Compute commond speed from some model with current information
    cmd_speed = predict_from_model(data)
   
    curr_speed = float(data['speed'])

    # Update PID controller
    throttle = speed_pid.update(cmd_speed, curr_speed)

    # send throttle
```