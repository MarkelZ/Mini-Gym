from abc import ABC, abstractmethod

from pygame import Surface, Vector2
import pygame

from minigym.gym_object.goal import Goal
from minigym.gym_object.hazard import Hazard
from minigym.physics.engine import PhysicsEngine
from minigym.agent.point_robot import PointRobot


class Environment(ABC):
    def __init__(self):
        self.physics = PhysicsEngine()
        self.hazards: list[Hazard] = []
        self.goal = Goal(Vector2())
        self.robot = PointRobot(Vector2(), self)

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

    def draw(self, surface: Surface) -> None:
        surface.fill((180, 180, 180))
        S = 100
        for i in range(4):
            for j in range(6):
                pygame.draw.rect(
                    surface,
                    (220, 220, 220),
                    pygame.Rect((i * 2 + (j % 2)) * S, j * S, S, S),
                )

        for h in self.hazards:
            h.draw(surface)

        self.goal.draw(surface)
        self.robot.draw(surface)
