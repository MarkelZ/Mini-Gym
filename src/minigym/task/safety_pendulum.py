from math import sin, cos

from pygame import Vector2

from minigym.agent.pendulum_robot import PendulumRobot
from minigym.task.environment import Environment


class SafetyPendulum(Environment):
    def __init__(self):
        super().__init__()

        self.robot: PendulumRobot = PendulumRobot(self, Vector2(0, 0))

    def act(self, throttle):
        self.robot.act(throttle)


    def cost(self):
        s = (1 - sin(self.robot.angle))/2 # How much is it pointing down? 0=upright, 1=down
        bad = s >= 0.5 # "Bad" if pointing halfway down
        return s + (10 if bad else 0)

    def reward(self):
        return -1e-3 * (self.robot.pos.x)**2 # Encourage x=0
