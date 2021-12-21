import argparse
import torch

from agents.action import train_model

def argparser():
    parser = argparse.ArgumentParser(description='MovieLens-100K Movie Recommendation with RL Agents')
    
    parser.add_argument('--gpu', type=str, default='0',
                        help='number of gpu device id')
    parser.add_argument('--agent', type=str, default='Q-FA', choices=['Q-FA', 'SARSA-FA', 'ActorCritic', 'REINFORCE', 'REINFORCE-Baseline','PPO', 'DDPG'],
                        help='an DRL algorithm used for agent')
    parser.add_argument('--time_steps', type=int, default=10000)

    return parser.parse_args()

def main():
    args = argparser()

    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    train_model(args, device)

if __name__ == "__main__":
    main()
