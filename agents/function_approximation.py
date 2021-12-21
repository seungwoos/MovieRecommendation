import numpy as np

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

class LinearFunctionApproximation():
    def __init__(self, env):
        self.env = env
        self.set_featurizer()
    
    def set_featurizer(self):
        observation_examples = np.array([self.env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()

        self.featurizer = sklearn.pipeline.FeatureUnion([
            ('rbf1', RBFSampler(gamma = 5.0, n_components = 20)),
            ('rbf2', RBFSampler(gamma = 2.0, n_components = 10)),
            ('rbf3', RBFSampler(gamma = 1.0, n_components = 10)),
            ('rbf4', RBFSampler(gamma = 0.5, n_components = 10))
        ])

        self.featurizer.fit(self.scaler.fit_transform(observation_examples))

    def featurize_state(self, state):
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized

class QLearningFA(LinearFunctionApproximation):
    def __init__(self, env, stats, epsilon=0.1, alpha=0.01, gamma=1.0):
        super(QLearningFA, self).__init__(env)

        self.stats = stats

        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.n
        
        self.w = np.zeros((self.n_action, 50))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def Q(self, state, action=None):
        if action is None:
            value = state.dot(self.w)
        else:
            value = state.dot(self.w[action])

        return value
        
    def get_action(self, state):
        probs = np.ones(self.n_action, dtype=float) * self.epsilon / self.n_action
        best_action =  np.argmax([self.Q(state, action) for action in range(self.n_action)])

        probs[best_action] += (1.0 - self.epsilon)
        action = np.random.choice(self.n_action, p=probs)
        return action

    def update_epsilon(self, epsilon):
        self.epsilon = np.min([epsilon, 0.1])
        
    def update_alpha(self, alpha):
        self.alpha = np.min([alpha, 0.01]) 

    def train(self, total_timesteps):

        for i in range(int(total_timesteps)):
            _state = self.env.reset()
            state = self.featurize_state(_state)

            done = False
            while not done:

                action = self.get_action(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = self.featurize_state(next_state)

                td_target = reward + self.gamma * np.max(self.Q(next_state))
                td_error = self.Q(state, action) - td_target

                dw = (td_error).dot(state)

                self.w[action] -= self.alpha * dw

                state = next_state
                self.stats.episode_rewards[i] += reward
                self.stats.episode_lengths[i] = self.env.local_step_counter

                if self.env.local_step_counter % 100 == 0:
                    print(f'[epoch {i}] - local step {self.env.local_step_counter} / {self.env.max_step}, cumulative reward is {self.stats.episode_rewards[i]}')

        self.env.close()
    
        return self.stats


class SarsaFA(LinearFunctionApproximation):
    def __init__(self, env, stats, epsilon=0.1, alpha=0.01, gamma=1.0):
        super(SarsaFA, self).__init__(env)

        self.stats = stats

        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.n
        
        self.w = np.zeros((self.n_action, 50))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def Q(self, state, action):
        value = state.dot(self.w[action])
        return value
        
    def get_action(self, state):
        probs = np.ones(self.n_action, dtype=float) * self.epsilon / self.n_action
        best_action =  np.argmax([self.Q(state, action) for action in range(self.n_action)])

        probs[best_action] += (1.0 - self.epsilon)
        action = np.random.choice(self.n_action, p=probs)

        return action

    def update_epsilon(self, epsilon):
        self.epsilon = np.min([epsilon, 0.1])
        
    def update_alpha(self, alpha):
        self.alpha = np.min([alpha, 0.01]) 

    def train(self, total_timesteps):

        for i in range(int(total_timesteps)):
            _state = self.env.reset()
            state = self.featurize_state(_state)

            done = False
            while not done:

                action = self.get_action(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = self.featurize_state(next_state)

                next_action = self.get_action(next_state)

                td_target = reward + self.gamma * self.Q(next_state, next_action)
                td_error = self.Q(state, action) - td_target

                dw = (td_error).dot(state)

                self.w[action] -= self.alpha * dw

                state = next_state
                self.stats.episode_rewards[i] += reward
                self.stats.episode_lengths[i] = self.env.local_step_counter

                if self.env.local_step_counter % 100 == 0:
                    print(f'[epoch {i}] - local step {self.env.local_step_counter} / {self.env.max_step}, cumulative reward is {self.stats.episode_rewards[i]}')

        self.env.close()
    
        return self.stats
