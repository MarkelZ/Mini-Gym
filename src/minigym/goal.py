from pygame import Surface, Vector2
import pygame

from minigym.geometry import Circle


class Goal:
    geom: Circle
    color: list[int] = [0, 255, 0]
    RADIUS = 40

    def __init__(self, pos: Vector2):
        self.geom = Circle(pos, self.RADIUS)

    @property
    def pos(self):
        return self.geom.center

    @pos.setter
    def pos(self, value: Vector2):
        self.geom.center = value

    def draw(self, surface: Surface):
        pygame.draw.circle(surface, self.color, self.geom.center, self.geom.radius)
