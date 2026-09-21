from math import cos, sin, pi

from pygame import Vector2
from minigym.physics.geometry import Geometry, LineSegment
from minigym.util import angle_to_vec2


class LidarBeam:
    geom: LineSegment
    _pos: Vector2
    _angle: float
    _length: float

    def __init__(self, pos: Vector2, angle: float, length: float = 200):
        self.geom = LineSegment(pos, Vector2())
        self._angle = angle
        self._length = length
        self._pos = pos
        self.update_geom()

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value: Vector2):
        self._pos = value
        self.update_geom()

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value: float):
        self._angle = value
        self.update_geom()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value: float):
        self._length = value
        self.update_geom()

    def update_geom_params(
        self, pos: Vector2 = None, angle: float = None, length: float = None
    ):
        if pos is not None:
            self._pos = pos

        if angle is not None:
            self._angle = angle

        if length is not None:
            self._length = length

        self.update_geom()

    def update_geom(self):
        self.geom.start = self._pos
        self.geom.end = self._pos + self._length * Vector2(
            cos(self._angle), sin(self._angle)
        )


class LidarBatch:
    _beams: list[LidarBeam]

    def __init__(self, num_beams, length):
        self._beams = [LidarBeam(Vector2(), 0, length) for _ in range(num_beams)]

    @property
    def num_beams(self):
        return len(self._beams)

    def update_orientation(self, origin: Vector2, angle: float, start_offset: float):
        theta_offset = (2 * pi) / (self.num_beams)
        for i in range(self.num_beams):
            theta = angle + i * theta_offset
            pos = origin + angle_to_vec2(theta, start_offset)
            self._beams[i].update_geom_params(pos=pos, angle=theta)

    def obs(self, target: Geometry) -> list[float]:
        intersects = [Geometry.intersection_point(b.geom, target) for b in self._beams]
        lidar_dist = [
            (1.0 - (p - b.pos).length() / b.length if p is not None else 0.0)
            for p, b in zip(intersects, self._beams)
        ]
        return lidar_dist
