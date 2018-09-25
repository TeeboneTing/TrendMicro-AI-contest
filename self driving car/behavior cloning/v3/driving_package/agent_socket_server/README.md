# Agent Socket Server

## Usage

Necessary to register 2 callback function,  
1. on_telemetry: Whenever
2. on_connect: 

The callback function prototype must be:

Args: data
Returns: Tupple of (throttle, steer_angle[degree])

```python
ai_server = AIAgentSocketServer()

def on_telemetry(data):
    print ('Speed: %s, Steering angle: %s' % (data['speed'], data['steering_angle']))

    throttle = 1.0
    steer_angle = 0
    return throttle, steer_angle

def on_reset(data):
    print ('reset')
    return 0, 0

ai_server.register_on_telemetry_callback(on_telemetry)
ai_server.register_on_connect_callback(on_reset)
ai_server.start()

```