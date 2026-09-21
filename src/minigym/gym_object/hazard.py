from pygame import Surface, Vector2
import pygame

from minigym.physics.geometry import Rectangle, Circle


class Hazard:
    geom: Circle
    color: list[int] = [64, 64, 255]
    SIZE = 40

    def __init__(self, pos: Vector2):
        self.geom = Circle(pos, self.SIZE)

    def draw(self, surface: Surface):
        # pygame.draw.rect(surface, self.color, self.geom.to_pygame_rect())
        pygame.draw.circle(surface, self.color, self.geom.center, self.geom.radius)

    @property
    def pos(self):
        return self.geom.center

    @pos.setter
    def pos(self, value: Vector2):
        self.geom.center = value

