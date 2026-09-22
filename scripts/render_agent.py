"""Load a saved agent and run it for one trajectory in the environment."""

import numpy as np
import torch
import pygame
import sys

from minigym.render.pygame_renderer import PygameRenderer
from minigym.task.safety_point_goal import SafetyPointGoal0
from minigym.algorithm.ppolag import load_checkpoint


PATH = "checkpoints/epoch_50.pt"
STEPS = 1000
DETERMINISTIC = True
DELTAT = 1. / 60.

ac, ckpt = load_checkpoint(PATH)
low = np.array(ckpt["action_low"], dtype=np.float32)
high = np.array(ckpt["action_high"], dtype=np.float32)

env = SafetyPointGoal0()
total_reward, total_cost = 0.0, 0.0
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
step = 0
while running:
    # Maintain the frame rate
    clock.tick(FPS)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    if step >= STEPS:
        continue

    step += 1

    # Update game logic
    obs = torch.as_tensor(np.array(env.obs(), dtype=np.float32))
    act = ac.act(obs, deterministic=DETERMINISTIC)
    act = np.clip(act, low, high)

    env.act(act.tolist())
    env.step(DELTAT)

    # Draw
    screen.fill((255, 0, 255))

    renderer.render(screen)

    # Update the display
    pygame.display.flip()


# Quit Pygame
pygame.quit()
sys.exit()