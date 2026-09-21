from pygame import Vector2

from minigym.physics.geometry import Circle


class Hazard:
    geom: Circle
    SIZE = 40

    def __init__(self, pos: Vector2):
        self.geom = Circle(pos, self.SIZE)

    @property
    def pos(self):
        return self.geom.center

    @pos.setter
    def pos(self, value: Vector2):
        self.geom.center = value

