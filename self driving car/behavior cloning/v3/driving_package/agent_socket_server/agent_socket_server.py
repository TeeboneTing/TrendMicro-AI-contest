from flask import Flask
import socketio
import eventlet.wsgi


class AIAgentSocketServer:
    def __init__(self):
        self.sio = socketio.Server()
        self.app = socketio.Middleware(self.sio, Flask(__name__))

        self.register_on_connect_callback(self.__default_on_connect)
        self.register_on_connect_callback(self.__default_on_telemetry)

    def __default_on_telemetry(self, data):
        return 0, 0
    
    def __default_on_connect(self, data):
        return 0, 0

    

    def register_on_telemetry_callback(self, callback_func):
        """
        Args:
            callback_func: A function handler must have input args: (data),
                and must return tuple (throttle, steering_angle_degree)
        """
        def control(sid, data):
            throttle, steering_angle_degree = callback_func(data)
            if throttle == 32767 and steering_angle_degree == 32767:
                self.send_restart()
                return

            self.sio.emit('steer', {
                'steering_angle': str(steering_angle_degree),
                'throttle': str(throttle)
            }, skip_sid=True)
        self.sio.on('telemetry', control)

    def register_on_connect_callback(self, callback_func):
        """
        Args:
            callback_func: A function handler must have input args: (data),
                and must return tuple (throttle, steering_angle_degree)
        """
        def reset(sid, data):
            callback_func(data)

            self.sio.emit('steer', {
                'steering_angle': str(0),
                'throttle': str(0)
            }, skip_sid=True)

        self.sio.on('connect', reset)

    def send_restart(self):
        self.sio.emit(
            "restart",
            data={},
            skip_sid=True)

    def start(self):
        eventlet.wsgi.server(eventlet.listen(('', 4567)), self.app)


if __name__ == "__main__":
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
