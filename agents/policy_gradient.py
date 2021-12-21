import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

class REINFORCE(nn.Module):
    def __init__(self, env, stats, learning_rate = 0.0002, gamma = 0.98):
        super(REINFORCE, self).__init__()

        self.env = env
        self.stats = stats

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.linear_1 = nn.Linear(env.observation_space.shape[0], 64)
        self.linear_2 = nn.Linear(64, env.action_space.n)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.data = []
        
    def policy(self, state):
        x = F.relu(self.linear_1(state))
        prob = F.softmax(self.linear_2(x), dim=0)
        return prob
    
    def get_action(self, state):
        prob = self.policy(torch.from_numpy(state).float())
        dist = Categorical(prob)
        action = dist.sample()
        return action.item(), prob
      
    def save(self, item):
        self.data.append(item)
        
    def update(self):
        G = 0
        self.optimizer.zero_grad()
        for reward, prob in self.data[::-1]:
            G = reward + self.gamma * G
            loss = - torch.log(prob) * G
            loss.backward()
        self.optimizer.step()
        self.data = []

    def train(self, total_timesteps):

        for i in range(total_timesteps):
            G = 0.0

            state = self.env.reset()
            done = False
            
            while not done:
                action, prob = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.save((reward, prob[action]))
                state = next_state
                G += reward

                self.stats.episode_rewards[i] += reward
                self.stats.episode_lengths[i] = self.env.local_step_counter

                if self.env.local_step_counter % 100 == 0:
                    avg_return = G / 100
                    print(f'[epoch {i}] - local step {self.env.local_step_counter} / {self.env.max_step}, cumulative reward is {self.stats.episode_rewards[i]} and average return is {avg_return}')

            self.update()

        self.env.close()
        return self.stats

class REINFORCE_B(REINFORCE):
    def __init__(self, env, stats, learning_rate = 0.0002):
        # super(REINFORCE, self).__init__()
        REINFORCE.__init__(self, env, stats)

        self.vf1 = nn.Linear(env.observation_space.shape[0], 64)
        self.vf2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)

        self.value_optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.data = []

    def get_value(self, x):
        x = torch.tensor(x).float()
        x = F.relu(self.vf1(x))
        x = F.relu(self.vf2(x))
        return self.output(x)
    
    def update_policy(self):
        G = 0
        self.optimizer.zero_grad()
        for reward, prob, v_t in self.data[::-1]:
            G = reward + self.gamma * G
            adv = G - v_t
            policy_loss = - torch.log(prob) * adv
            policy_loss.backward()
        self.optimizer.step()
        self.data = []
    
    def update_value(self):
        G = 0
        self.value_optimizer.zero_grad()
        for reward, prob, v_t in self.data[::-1]:
            G = reward + self.gamma * G
            adv = G - v_t
            value_loss = torch.mean(adv)
            value_loss.backward()
        self.value_optimizer.step()
        self.data = []

    def train(self, total_timesteps):

        for i in range(total_timesteps):
            G = 0.0

            state = self.env.reset()
            done = False
            
            while not done:
                action, prob = self.get_action(state)
                v_t = self.get_value(state)
                next_state, reward, done, _ = self.env.step(action)
                self.save((reward, prob[action], v_t))
                state = next_state
                G += reward

                self.stats.episode_rewards[i] += reward
                self.stats.episode_lengths[i] = self.env.local_step_counter

                if self.env.local_step_counter % 100 == 0:
                    avg_return = G / 100
                    print(f'[epoch {i}] - local step {self.env.local_step_counter} / {self.env.max_step}, cumulative reward is {self.stats.episode_rewards[i]} and average return is {avg_return}')

            self.update_value()
            self.update_policy()

        self.env.close()
        return self.stats


class ActorCritic(nn.Module):
    def __init__(self, env, stats, gamma=0.98, learning_rate=0.0002):
        super(ActorCritic, self).__init__()

        self.env = env
        self.stats = stats

        print(env.observation_space.shape[0])
        self.shared_layer = nn.Linear(env.observation_space.shape[0], 256)
        self.policy_layer = nn.Linear(256, env.action_space.n)
        self.value_layer = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.data = []
        self.gamma = gamma
        self.learning_rate=learning_rate
        
    def policy(self, state, softmax_dim = 0):
        x = F.relu(self.shared_layer(state))
        x = self.policy_layer(x)
        prob = F.softmax(x, dim=softmax_dim)
        return Categorical(prob)
    
    def get_action(self, state):
        dist = self.policy(torch.from_numpy(state).float())
        action = dist.sample().item()
        return action
    
    def value(self, state):
        x = F.relu(self.shared_layer(state))
        value = self.value_layer(x)
        return value
    
    def save(self, transition):
        self.data.append(transition)
        
    def get_batch(self):
        state_list, action_list, reward_list, next_state_list, done_list = [], [], [], [], []
        for experience in self.data:
            state, action, reward, next_state, done = experience
            state_list.append(state)
            action_list.append([action])
            reward_list.append([reward/100.0])
            next_state_list.append(next_state)
            done_list.append([0.0 if done else 1.0])
        
        state_batch = torch.tensor(state_list, dtype=torch.float)
        action_batch = torch.tensor(action_list)
        reward_batch = torch.tensor(reward_list, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_list, dtype=torch.float)
        done_batch = torch.tensor(done_list, dtype=torch.float)
        self.data = []

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
  
    def update(self):
        state, action, reward, next_state, done = self.get_batch()
        td_target = reward + gamma * self.value(next_state) * done
        td_error = td_target - self.value(state)
        
        dist = self.policy(state, softmax_dim=1)
        prob_action = dist.probs.gather(1, action)
        loss_actor = - torch.log(prob_action) * td_error.detach()

        loss_critic = (self.value(state) - td_target.detach())**2
        loss = loss_actor + loss_critic

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def train(self, total_timesteps, update_every=10, test=False):
        for i in range(total_timesteps):
            done = False
            state = self.env.reset()

            while not done:
                for t in range(update_every):
                    action = self.get_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    self.save((state, action, reward, next_state,done))
                    state = next_state

                    self.stats.episode_rewards[i] += reward
                    self.stats.episode_lengths[i] = self.env.local_step_counter

                    if done:
                        break                     
                self.update()

        self.env.close()
        return self.stats
