from pygame import Vector2
from math import pi

from minigym.physics.kinetic_pointmass import KineticPointmass
from minigym.physics.geometry import Circle, Geometry
from minigym.agent.sensor.lidar import LidarBeam
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

    def __init__(self, pos: Vector2, env):
        from minigym.task.environment import Environment

        self.env: Environment = env

        self.pointmass = KineticPointmass(pos)
        self.pointmass.friction = self._LINEAR_FRICTION
        self.env.physics.add_pointmass(self.pointmass)
        self.geom = Circle(pos, self.RADIUS)

        self.hazard_lidars: list[LidarBeam] = [
            LidarBeam(Vector2(), 0) for _ in range(self.NUM_HAZARD_LIDARS)
        ]
        self.goal_lidars: list[LidarBeam] = [
            LidarBeam(Vector2(), 0) for _ in range(self.NUM_HAZARD_LIDARS)
        ]
        self._update_lidar_pos()

    def _update_lidar_pos(self):
        theta = (2 * pi) / (self.NUM_HAZARD_LIDARS)
        for i in range(self.NUM_HAZARD_LIDARS):
            angle = self.angle + i * theta
            pos = self.pos + angle_to_vec2(angle, self.RADIUS)
            self.hazard_lidars[i].update_geom_params(pos=pos, angle=angle)

        theta = (2 * pi) / (self.NUM_GOAL_LIDARS)
        for i in range(self.NUM_GOAL_LIDARS):
            angle = self.angle + i * theta
            pos = self.pos + angle_to_vec2(angle, self.RADIUS)
            self.goal_lidars[i].update_geom_params(pos=pos, angle=angle)

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
        hazard_intersects = [
            Geometry.intersection_point(
                self.hazard_lidars[i].geom, self.env.hazards[0].geom
            )
            for i in range(self.NUM_HAZARD_LIDARS)
        ]
        hazard_lidar_dist = [
            (
                1
                - (hazard_intersects[i] - self.hazard_lidars[i].pos).length()
                / self.hazard_lidars[i]._length
                if hazard_intersects[i] is not None
                else 0.0
            )
            for i in range(self.NUM_HAZARD_LIDARS)
        ]
        goal_intersects = [
            Geometry.intersection_point(self.goal_lidars[i].geom, self.env.goal.geom)
            for i in range(self.NUM_GOAL_LIDARS)
        ]
        goal_lidar_dist = [
            (
                1
                - (goal_intersects[i] - self.goal_lidars[i].pos).length()
                / self.goal_lidars[i]._length
                if goal_intersects[i] is not None
                else 0.0
            )
            for i in range(self.NUM_GOAL_LIDARS)
        ]
        return (
            hazard_lidar_dist
            + goal_lidar_dist
            # + [self.angle]
            # + [self.vel.x, self.vel.y]
        )

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
        self._update_lidar_pos()

