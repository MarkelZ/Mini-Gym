from abc import ABC, abstractmethod

from minigym.agent.robot import Robot
from minigym.physics.engine import PhysicsEngine


class Environment(ABC):
    def __init__(self):
        self.physics = PhysicsEngine()
        self.robot: Robot = None

    def obs(self) -> list[float]:
        return self.robot.obs()

    def cost(self) -> float:
        return 0

    def reward(self) -> float:
        return 0

    @abstractmethod
    def act(self, args: list[float]) -> None:
        pass

    def step(self, deltat: float) -> None:
        self.physics.update(deltat)
        self.robot.update(deltat)
