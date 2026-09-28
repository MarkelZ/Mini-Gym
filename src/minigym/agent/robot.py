

from abc import abstractmethod

from pygame import Vector2

from minigym.physics.geometry import Geometry


class Robot:
    geom: Geometry

    def __init__(self, env):
        from minigym.task.environment import Environment

        self.env: Environment = env

    @abstractmethod
    def act(self, *args):
        pass

    def obs(self) -> list[float]:
        return []

    @abstractmethod
    def update(self, deltat: float):
        pass