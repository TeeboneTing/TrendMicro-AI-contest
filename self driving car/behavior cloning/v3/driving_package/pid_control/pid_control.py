import time


class PIDController:
    def __init__(self, Kp, Ki, Kd, max_I=None, max_output=None):

        if any([Kp < 0, Ki < 0, Kd < 0]):
            raise Exception('Invalid negative parameter')

        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd

        self._max_I = 2147483647 if max_I is None else abs(max_I)
        self._max_output = 2147483647 if max_output is None else abs(max_output)

        self._error = 0
        self._prev_error = 0
        self._integral_error = 0
        self._last_update_time = None

        self._p_gain = 0
        self._i_gain = 0
        self._d_gain = 0

    def reset(self):
        self._error = 0
        self._prev_error = 0
        self._integral_error = 0
        self._last_update_time = None

        self._p_gain = 0
        self._i_gain = 0
        self._d_gain = 0

    @property
    def Kp(self):
        return self._Kp

    @Kp.setter
    def Kp(self, Kp):
        if Kp < 0:
            raise Exception('Invalid negative parameter')
        self._Kp = Kp

    @property
    def Ki(self):
        return self._Kp

    @Ki.setter
    def Ki(self, Ki):
        if Ki < 0:
            raise Exception('Invalid negative parameter')
        self._Ki = Ki

    @property
    def Kd(self):
        return self._Kd

    @Kd.setter
    def Kd(self, Kd):
        if Kd < 0:
            raise Exception('Invalid negative parameter')
        self._Kd = Kd

    @property
    def p_gain(self):
        return self._p_gain

    @property
    def i_gain(self):
        return self._i_gain

    @property
    def d_gain(self):
        return self._d_gain

    @property
    def error(self):
        return self._error

    def update(self, target, current):
        now = time.time()
        delta_time = None
        if self._last_update_time:
            delta_time = now - self._last_update_time

        self._error = target - current
        self._p_gain = self._Kp * self._error

        if delta_time:
            self._integral_error += self._error * delta_time
        self._i_gain = self._Ki * self._integral_error
        self._i_gain = min(max(self._i_gain, -self._max_I), self._max_I)

        d_value = 0
        if delta_time:
            d_value = (self._error - self._prev_error) / delta_time
        self._d_gain = self._Kd * d_value

        self._prev_error = self._error
        self._last_update_time = now
        output = self._p_gain + self._i_gain + self._d_gain

        return min(max(output, -self._max_output), self._max_output)


if __name__ == '__main__':
    pid = PIDController(0.1, 0.1, 0.1)

    target = 10
    current = 0

    import random
    while(1):
        time.sleep(0.1)
        print ('out', pid.update(target, current))
        print ('p_gain', pid.p_gain)
        print ('i_gain', pid.i_gain)
        print ('d_gain', pid.d_gain)

        if current < target:
            current += random.random()
        elif current > target:
            current -= random.random()
