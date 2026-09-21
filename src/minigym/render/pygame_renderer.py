import pygame

from minigym.task.environment import Environment
from minigym.util import angle_to_vec2


class PygameRenderer:
    # Appearance
    goal_color: list[int] = [0, 255, 0]
    hazard_color: list[int] = [64, 64, 255]
    robot_color1: list[int] = [255, 0, 0]
    robot_color2: list[int] = [0, 0, 255]
    robot_direction_width = 5

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

    def draw_robot(self, surface: pygame.Surface) -> None:
        robot = self.env.robot

        # Visualize cost and reward
        if self.env.cost() > 0:
            pygame.draw.circle(surface, (128, 0, 0), robot.pos, robot.geom.radius * 2)
        elif self.env.reward() > 0:
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
            l = robot.hazard_lidars[i]
            o_h = robot.obs()[i]
            o_g = robot.obs()[i + robot.NUM_GOAL_LIDARS]
            color = (0, 255 * o_g, 255 * o_h)
            pygame.draw.circle(surface, color, l.pos + angle_to_vec2(l._angle, 10), 5)

    def render(self, surface: pygame.Surface) -> None:
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

        # Gym objects
        self.draw_hazards(surface)
        self.draw_goal(surface)
        self.draw_robot(surface)
