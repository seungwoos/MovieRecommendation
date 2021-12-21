# Codes from https://github.com/dennybritz/reinforcement-learning/blob/master/lib/plotting.py 

import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

def plot_episode_stats(stats, save_path, smoothing_window=20):
    
    # Plot the episode reward over time
    fig = plt.figure(figsize=(10,5))
    print(stats.episode_rewards)
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    print(rewards_smoothed)
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    fig.savefig('fig1.png')
