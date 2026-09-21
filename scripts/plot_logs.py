"""Plot reward and cost per epoch from a PPO-Lagrangian training log."""

import argparse
import csv

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", nargs="?", default="logs/train_log.csv")
    parser.add_argument("--out", default="logs/training_curves.png")
    args = parser.parse_args()

    epochs, rewards, costs = [], [], []
    with open(args.log_path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            rewards.append(float(row["avg_reward"]))
            costs.append(float(row["avg_cost"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    ax1.plot(epochs, rewards)
    ax1.set_ylabel("avg reward / episode")
    ax1.grid(True)

    ax2.plot(epochs, costs, color="tab:red")
    ax2.set_ylabel("avg cost / episode")
    ax2.set_xlabel("epoch")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"saved plot to {args.out}")
