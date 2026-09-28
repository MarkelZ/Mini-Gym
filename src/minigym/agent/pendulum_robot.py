

from pygame import Vector2

from minigym.agent.robot import Robot
from minigym.physics.kinetic_pointmass import KineticPointmass
from minigym.physics.link import Link
from minigym.util import clamp, points_to_angle


class PendulumRobot(Robot):
    POLE_LENGTH = 200
    GRAVITY = 80
    THROTTLE_MUL = 5000

    def __init__(self, env, pos: Vector2):
        super().__init__(env)

        self.body = KineticPointmass(pos)
        self.tip = KineticPointmass(pos - Vector2(0, self.POLE_LENGTH))
        self.body.friction = 0.95
        self.body.mass = 2
        self.tip.mass = 0.1
        self.tip.friction = 1.0
        self.link = Link(self.body ,self.tip, length=self.POLE_LENGTH)

        self.env.physics.add_pointmass(self.body)
        self.env.physics.add_pointmass(self.tip)
        self.env.physics.add_link(self.link)

    @property
    def angle(self):
        return points_to_angle(self.body.pos, self.tip.pos)

    @property
    def pos(self):
        return self.body.pos

    def act(self, throttle: float):
        v = clamp(throttle, -1, 1)
        self.body.apply_force(Vector2(v * self.THROTTLE_MUL, 0))

    def update(self, deltat: float):
        self.tip.apply_force(Vector2(0, self.GRAVITY))
        self.body._pos.y = 0
