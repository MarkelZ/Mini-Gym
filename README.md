# Mini-Gym

A minimalist [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) clone.
We avoid running [MuJoCo](https://mujoco.org/) altogether and replace it with a more basic physics engine to dramatically speed up rollout time.

https://github.com/user-attachments/assets/113b18a5-d7d9-4ffb-a050-2ebfc305877b

## Install and run

Install with
```
pip3 install -e .
```

Run script files in `Mini-Gym/scripts/`.

* `manual_control.py`: interactively control the robot yourself with WASD.
* `measure_rollout_tps.py`: measures rollout time.
* `plot_logs.py`: plots the logs generated after training an agent.
* `render_agent.py`: visualizes a trajectory of a (partially) trained agent.
* `rollout_trained_agent.py`: roll out a (partially) trained agent to get a trajectory.
* `train.py`: train an agent.

## Physics

Custom minimal engine with Euler integration.

## License

This code is licensed under the terms of the MIT license.
