import pygame
import sys

from minigym.environment import SafetyPointGoal0

# Initialize Pygame
pygame.init()

# Window settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Safety Gymnasium Clone")

# Clock controls the frame rate
clock = pygame.time.Clock()
FPS = 60

env = SafetyPointGoal0()

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
    screen.fill((180, 180, 180))

    S = 100
    for i in range(4):
        for j in range(6):
            pygame.draw.rect(
                screen, (220, 220, 220), pygame.Rect((i * 2 + (j % 2)) * S, j*S, S, S)
            )

    env.draw(screen)

    # Update the display
    pygame.display.flip()

    # Maintain the frame rate
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
sys.exit()
