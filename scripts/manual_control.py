import pygame
import sys

from minigym.render.pygame_renderer import PygameRenderer
from minigym.task.safety_point_goal import SafetyPointGoal0

# Initialize Pygame
pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mini-Gym")

# Clock controls the frame rate
clock = pygame.time.Clock()
FPS = 60

env = SafetyPointGoal0()
renderer = PygameRenderer(env)

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Update game logic
    keys = pygame.key.get_pressed()

    throttle = 0
    steer = 0
    if keys[pygame.K_w]:
        throttle += 1
    if keys[pygame.K_a]:
        steer = -1
    if keys[pygame.K_d]:
        steer = 1

    env.act([throttle, steer])
    env.step(1 / FPS)

    # Draw
    screen.fill((255, 0, 255))

    renderer.render(screen)

    # Update the display
    pygame.display.flip()

    # Maintain the frame rate
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
sys.exit()
