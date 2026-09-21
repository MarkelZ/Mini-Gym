from time import time
from minigym.task.safety_point_goal import SafetyPointGoal0
from tqdm import tqdm


env = SafetyPointGoal0()

o = env.obs()

t0 = time()
N = int(1e6)
for _ in tqdm(range(N)):
    env.act([1, 1])
    env.step(0.1)
    o = env.obs()
t = time()

print(f"{N} simulation steps in {t-t0} seconds")
print(f"{(t-t0)*1000/N} ms per step")
