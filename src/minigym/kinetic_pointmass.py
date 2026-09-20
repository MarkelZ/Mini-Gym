from pygame import Vector2


class KineticPointmass:
    _pos: Vector2
    _vel: Vector2
    _acc: Vector2
    friction: float = 0.98
    mass: float = 1
    maxvel: float = 10
    maxacc: float = 4

    def __init__(self, pos: Vector2, vel: Vector2 = None):
        self._pos = pos
        self._vel = Vector2(0, 0) if vel is None else vel
        self._acc = Vector2(0, 0)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value: Vector2):
        self._pos = value

    @property
    def vel(self):
        return self._vel

    @pos.setter
    def vel(self, value: Vector2):
        self._vel = value

    def apply_force(self, force: Vector2):
        self._acc += force / self.mass

    def apply_dir_force(self, force: float, dir: Vector2):
        n = dir.normalize()
        self.apply_force(n * force)
