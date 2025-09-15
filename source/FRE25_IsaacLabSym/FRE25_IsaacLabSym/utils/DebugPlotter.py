import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()
plt.ion()


class DebugPlotter:
    LIDAR_N_RAYS = 40

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.totalRewards = np.array([])

    def add_total_reward(self, totalRewards: torch.Tensor):
        self.totalRewards = np.append(self.totalRewards, totalRewards.cpu().numpy())

    def plot(self):
        self.ax.clear()
        sns.histplot(self.totalRewards, ax=self.ax, kde=True)
        self.ax.set_title("Total Rewards")
        self.ax.set_xlabel("Total Reward")
        self.ax.set_ylabel("Count")
        plt.draw()
        plt.pause(0.01)
