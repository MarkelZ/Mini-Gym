from pygame import Vector2
from math import pi

from minigym.physics.kinetic_pointmass import KineticPointmass
from minigym.physics.geometry import Circle, Geometry
from minigym.agent.sensor.lidar import LidarBatch, LidarBeam
from minigym.util import clamp, angle_to_vec2


class PointRobot:
    # Controls
    THROTTLE_MUL = 5000
    STEER_MUL = 50

    # Physics settings
    _LINEAR_FRICTION: float = 0.8
    _ANGULAR_FRICTION: float = 0.8

    # Geometry
    geom: Circle
    RADIUS = 30

    # LIDARS
    NUM_HAZARD_LIDARS = 8
    NUM_GOAL_LIDARS = 8
    LIDAR_LENGTH = 400

    # Physics
    pointmass: KineticPointmass

    # Angle, in radians
    angle: float = 0
    angular_acc: float = 0
    angular_vel: float = 0

    hazard_lidar: LidarBatch
    goal_lidar: LidarBatch

    def __init__(self, pos: Vector2, env):
        from minigym.task.environment import Environment

        self.env: Environment = env

        self.pointmass = KineticPointmass(pos)
        self.pointmass.friction = self._LINEAR_FRICTION
        self.env.physics.add_pointmass(self.pointmass)
        self.geom = Circle(pos, self.RADIUS)

        self.hazard_lidar = LidarBatch(self.NUM_HAZARD_LIDARS, self.LIDAR_LENGTH)
        self.goal_lidar = LidarBatch(self.NUM_GOAL_LIDARS, self.LIDAR_LENGTH)
        self.hazard_lidar.update_orientation(self.pos, self.angle, self.RADIUS)
        self.goal_lidar.update_orientation(self.pos, self.angle, self.RADIUS)

    @property
    def pos(self):
        return self.pointmass.pos

    @pos.setter
    def pos(self, value: Vector2):
        self.pointmass.pos = value

    @property
    def vel(self):
        return self.pointmass.vel

    @vel.setter
    def vel(self, value: Vector2):
        self.pointmass.vel = value

    def act(self, throttle: float, steer: float):
        clamp(throttle, 0, 1)
        clamp(steer, -1, 1)
        self.angular_acc += self.STEER_MUL * steer
        self.pointmass.apply_dir_force(
            throttle * self.THROTTLE_MUL, angle_to_vec2(self.angle)
        )

    def obs(self) -> list[bool]:
        goal_obs: list[float] = self.goal_lidar.obs(self.env.goal.geom)
        hazard_obs: list[float] = self.hazard_lidar.obs(self.env.hazards[0].geom)
        return hazard_obs + goal_obs

    def update(self, deltat: float):
        # Update angle with steer physics
        self.angular_vel = self._ANGULAR_FRICTION * (
            self.angular_vel + self.angular_acc * deltat
        )
        self.angle += self.angular_vel * deltat
        self.angular_acc = 0

        # Move geometry to physical location
        self.geom.center = self.pointmass.pos

        # Move LIDARs to physical location
        self.hazard_lidar.update_orientation(self.pos, self.angle, self.RADIUS)
        self.goal_lidar.update_orientation(self.pos, self.angle, self.RADIUS)
