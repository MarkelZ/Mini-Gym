from math import cos, sin

from pygame import Vector2
from minigym.geometry import LineSegment


class LidarBeam:
    geom: LineSegment
    origin: float
    angle: float
    length: float

    def __init__(self, pos: Vector2, angle: float):
        self.geom = LineSegment(pos, Vector2())
        self.angle = angle
        self.length = 200
        self.pos = pos
        self.update_geom()

    @property
    def pos(self):
        return self.geom.start

    @pos.setter
    def pos(self, value: Vector2):
        self.geom.start = value

    def update_geom(self):
        self.geom.end = self.pos + self.length * Vector2(cos(self.angle), sin(self.angle))
