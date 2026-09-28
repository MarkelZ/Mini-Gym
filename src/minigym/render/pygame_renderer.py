import pygame

from minigym.agent.pendulum_robot import PendulumRobot
from minigym.task.environment import Environment
from minigym.task.safety_pendulum import SafetyPendulum
from minigym.task.safety_point_goal import SafetyPointGoal0
from minigym.util import angle_to_vec2


class PygameRenderer:
    # Appearance
    # Safety point goal
    goal_color: list[int] = [0, 255, 0]
    hazard_color: list[int] = [64, 64, 255]
    robot_color1: list[int] = [255, 0, 0]
    robot_color2: list[int] = [0, 0, 255]
    robot_direction_width = 5
    # Pendulum
    robot_width = 100
    robot_height = 50
    pole_thickness = 10
    capped = True
    shift_robot = pygame.Vector2(320, 310)

    # The environment
    env: Environment

    def __init__(self, env: Environment):
        self.env = env

    def draw_goal(self, surface: pygame.Surface) -> None:
        g = self.env.goal
        pygame.draw.circle(surface, self.goal_color, g.geom.center, g.geom.radius)

    def draw_hazards(self, surface: pygame.Surface) -> None:
        for h in self.env.hazards:
            pygame.draw.circle(surface, self.hazard_color, h.geom.center, h.geom.radius)

    def draw_point_robot(self, surface: pygame.Surface) -> None:
        robot = self.env.robot

        # Visualize cost and reward
        if self.env.cost() > 0:
            pygame.draw.circle(surface, (128, 0, 0), robot.pos, robot.geom.radius * 2)
        elif self.env.reward() >= 1.0 - 1e-5:
            pygame.draw.circle(surface, (0, 128, 0), robot.pos, robot.geom.radius * 2)

        # Draw pos and angle
        pygame.draw.circle(surface, self.robot_color1, robot.pos, robot.geom.radius)
        pygame.draw.line(
            surface,
            self.robot_color2,
            robot.pos,
            robot.pos + angle_to_vec2(robot.angle, robot.RADIUS + 5),
            self.robot_direction_width,
        )

        # LIDARS, kinda hacky
        for i in range(robot.NUM_HAZARD_LIDARS):
            l = robot.hazard_lidar._beams[i]
            o_h = robot.obs()[i]
            o_g = robot.obs()[i + robot.NUM_GOAL_LIDARS]
            color = (0, 255 * o_g, 255 * o_h)
            pygame.draw.circle(surface, color, l.pos + angle_to_vec2(l._angle, 10), 5)
            # pygame.draw.line(surface, (255, 0, 255), l.geom.start, l.geom.end)

    def draw_pendulum_robot(self, surface: pygame.Surface) -> None:
        robot: PendulumRobot = self.env.robot
        pos = robot.body.pos + self.shift_robot
        tip = robot.tip.pos + self.shift_robot
        w2 = self.robot_width / 2
        h2 = self.robot_height / 2
        # Body
        pygame.draw.rect(
            surface,
            self.robot_color1,
            pygame.Rect(
                pos.x - w2,
                pos.y - h2,
                self.robot_width,
                self.robot_height,
            ),
        )

        # Pole
        pygame.draw.line(surface, self.robot_color2, pos, tip, self.pole_thickness)
        if self.capped:
            pygame.draw.circle(surface, self.robot_color2, pos, self.pole_thickness//2)
            pygame.draw.circle(surface, self.robot_color2, tip, self.pole_thickness//2)

    def draw_background(self, surface: pygame.Surface) -> None:
        # Background
        surface.fill((180, 180, 180))
        S = 100
        for i in range(4):
            for j in range(6):
                pygame.draw.rect(
                    surface,
                    (220, 220, 220),
                    pygame.Rect((i * 2 + (j % 2)) * S, j * S, S, S),
                )

    def render_safety_point_goal(self, surface: pygame.Surface) -> None:
        # Background
        self.draw_background(surface)

        # Gym objects
        self.draw_hazards(surface)
        self.draw_goal(surface)
        self.draw_point_robot(surface)

    def render_safety_pendulum(self, surface: pygame.Surface) -> None:
        # Background
        self.draw_background(surface)
        self.draw_pendulum_robot(surface)

    def render(self, surface: pygame.Surface) -> None:
        if isinstance(self.env, SafetyPointGoal0):
            self.render_safety_point_goal(surface)
        elif isinstance(self.env, SafetyPendulum):
            self.render_safety_pendulum(surface)
        else:
            raise NotImplementedError(f"Unsupported env type: {self.env}")
