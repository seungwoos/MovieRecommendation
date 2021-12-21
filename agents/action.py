import time
import gym
import numpy as np
import pandas as pd

from .function_approximation import QLearningFA, SarsaFA
from .policy_gradient import ActorCritic, REINFORCE, REINFORCE_B
from stable_baselines3 import PPO, DDPG
from .utils import EpisodeStats, plot_episode_stats

def train_model(args, device):
    env = gym.make('movie_env:RecoEnv-v0')
    env.reset()

    start_time = time.perf_counter()
    model = get_model(args, env, device)
    
    if args.agent in ['PPO', 'DDPG']:
        model.learn(total_timesteps=args.time_steps)
    else:
        stats = model.train(total_timesteps = args.time_steps)
    end_time = time.perf_counter()

    print(f'Training {args.agent} takes {end_time - start_time} seconds.')

    save_path = f'./Figures/{args.agent}.png'
    plot_episode_stats(stats, save_path, smoothing_window=25)


def get_model(args, env, device):

    stats = EpisodeStats(episode_lengths = np.zeros(args.time_steps), episode_rewards=np.zeros(args.time_steps))
    model = args.agent

    if model == 'Q-FA':
        return QLearningFA(env=env, stats=stats)
    elif model == 'SARSA-FA':
        return SarsaFA(env=env, stats=stats)
    elif model == 'ActorCritic':
        return ActorCritic(env=env, stats=stats)
    elif model == 'REINFORCE':
        return REINFORCE(env=env, stats=stats)
    elif model == 'REINFORCE-Baseline':
        return REINFORCE_B(env=env, stats=stats)
    elif model == 'PPO':
        return PPO(policy='MlpPolicy', env=env, verbose=1, device=device)
    elif model == 'DDPG':
        return DDPG(policy='MlpPolicy', env=env, verbose=1, device=device)
    else:
        raise NotImplemententedError
