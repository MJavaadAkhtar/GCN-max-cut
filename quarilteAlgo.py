
def findQ3(lst):
    M = Q1 = Q3 = lst[0]
    step =step_Q1 = step_Q3 = max(lst[0]/2, 0.1)
    step = max(lst[0]/2, 0.1)
    for data in lst :
        # update median M
        if M > data:
            M = M - step
        elif M < data:
            M = M + step
        if abs(data-M) < step:
            step = step /2

        if data > M:
            if Q3 > data:
                Q3 = Q3 - step_Q3
            elif Q3 < data:
                Q3 = Q3 + step_Q3
            if abs(data-Q3) < step_Q3:
                step_Q3 = step_Q3 /2
    print(M, Q1, Q3)

class RunningPercentile:
    def __init__(self, percentile=0.5, step=0.1):
        self.step = step
        self.step_up = 1.0 - percentile
        self.step_down = percentile
        self.x = None

    def push(self, observation):
        if self.x is None:
            self.x = observation
            return

        if self.x > observation:
            self.x -= self.step * self.step_up
        elif self.x < observation:
            self.x += self.step * self.step_down
        if abs(observation - self.x) < self.step:
            self.step /= 2.0

import numpy as np
import matplotlib.pyplot as plt
class RunningPercentile:
    def __init__(self, percentile=0.5, step=0.1):
        self.step = step
        self.step_up = 1.0 - percentile
        self.step_down = percentile
        self.x = None

    def push(self, observation):
        if self.x is None:
            self.x = observation
            return
        self.step = max(abs(observation), self.step)
        if self.x > observation:
            self.x -= self.step * self.step_up
        elif self.x < observation:
            self.x += self.step * self.step_down
        if abs(observation - self.x) < self.step:
            self.step /= 2.0

findQ3([10,5,6.7,5,6,5,5.8, 5, 5.5, 5])
