from abc import ABC, abstractmethod

from pygame import Surface, Vector2

from minigym.geometry import Geometry
from minigym.goal import Goal
from minigym.hazard import Hazard
from minigym.physics_engine import PhysicsEngine
from minigym.robot import PointRobot


class Environment(ABC):
    def __init__(self):
        self.physics = PhysicsEngine()
        self.hazards: list[Hazard] = []
        self.goal = Goal(Vector2())
        self.robot = PointRobot(Vector2(), self)

    def obs(self):
        return self.robot.obs()

    def cost(self):
        return 0

    def reward(self):
        return 0

    @abstractmethod
    def act(self, args: list[float]):
        pass

    def step(self, deltat: float):
        self.physics.update(deltat)
        self.robot.update(deltat)

    def draw(self, surface: Surface):
        for h in self.hazards:
            h.draw(surface)
        self.goal.draw(surface)
        self.robot.draw(surface)


class SafetyPointGoal0(Environment):
    def __init__(self):
        super().__init__()
        self.robot.pos = Vector2(80, 80)
        self.hazards.append(Hazard(Vector2(250, 200)))
        self.goal.pos = Vector2(400, 500)

    def act(self, args: list[float]):
        self.robot.act(args[0], args[1])

    def cost(self):
        # For quantitative instead of qualitative cost, use Geometry.intersection_point
        collision = any(
            [Geometry.intersects_with(self.robot.geom, h.geom) for h in self.hazards]
        )
        if collision:
            return 1
        else:
            return 0

    def reward(self):
        win = Geometry.intersects_with(self.robot.geom, self.goal.geom)
        if win:
            return 1
        else:
            return 0
