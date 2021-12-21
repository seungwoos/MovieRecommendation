import numpy as np
import math
from gym import spaces
import gym
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

class MovieRecoEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self, user_df: pd.DataFrame, item_df: pd.DataFrame, data_df: pd.DataFrame, 
                    n_movies: int = 50, seed: int = 42):

        self.user_df = user_df
        self.item_df = item_df
        self.data_df = data_df

        self.n_movies = n_movies
        self._random_state = np.random.RandomState(seed)
        self.seed = seed

        self.users = self._get_users()
        self.movies = self._get_movies()
        self.data = self._get_data()

        self.user_mean = self.data.groupby('user_id').mean().to_dict()['rating']
        self.movie_mean = self.data.groupby('movie_id').mean().to_dict()['rating']

        self.local_step_counter = 0
        self.max_step = self.data.shape[0] - 1
        
        self.reward = 0.0
        self.observation = None
        self.action = 0
        self.done = False

        self.action_space = spaces.Discrete(n_movies)
        self.observation_space = spaces.Box(low=0., high=1., shape=self._get_observation().shape, dtype=np.float32)

    def step(self, action):
        
        self.reward = self._get_reward(action)
        self.observation = self._get_observation()

        if self.local_step_counter == self.max_step:
            self.done = True
        
        self.local_step_counter += 1

        return self.observation, self.reward, self.done, {}

    def _get_observation(self):
        user_id, movie_id = self._get_user_movie_id()

        user_mean = np.array([self.user_mean.get(user_id, 3.) / 5,], dtype=np.float32)
        movie_mean = np.array([self.movie_mean.get(movie_id, 3.) / 5.], dtype=np.float32)

        user_info = self.users.loc[self.users['user_id'] == user_id].drop(['user_id'], axis=1).to_numpy()

        return np.concatenate([user_mean, movie_mean, user_info[0]])
    
    def reset(self):
        self.users = self._get_users()
        self.movies = self._get_movies()
        self.data = self._get_data()

        self.local_step_counter = 0
        self.reward = 0.0
        self.observation = None
        self.action = 0
        self.done = False

        return self._get_observation()

    def render(self, mode='human'):
        return self._get_observation()

    def _get_users(self):
        user = pd.read_csv(self.user_df, header=None, sep= "|", names=["user_id", "age", "gender", "occupation", "zipcode"])
        # user.columns = ["user_id", "age", "gender", "occupation", "zipcode"]

        bins = [0, 20, 30, 40, 50, 60, np.inf]
        names = ['<20', '20-29', '30-39','40-49', '51-60', '60+']

        user['age_group'] = pd.cut(user['age'], bins, labels=names)
        user = user.drop(["age"], axis = 1)

        feature_name = ['user_id', 'age_group', 'gender', 'occupation']
        user_features = user[user.columns.intersection(feature_name)]

        feature_name.remove('user_id')

        user_features[feature_name] = user_features[feature_name].apply(lambda x: pd.factorize(x)[0])
        user_features['occupation'] = user_features['occupation'] / user_features['occupation'].max()
        user_features['age_group'] = user_features['age_group'] / user_features['age_group'].max()

        # print(user_features[feature_name].min(), user_features[feature_name].max())
        # print(user_features.head())
        return user_features

    def _get_movies(self):
        movie = pd.read_csv(self.item_df, header = None, sep = "|", encoding='latin-1')
        movie.columns = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL", 
                        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

        movie_features = movie.drop(['movie_title', 'release_date', 'video_release_date', 'IMDb_URL'],axis = 1)
        return movie_features

    def _get_data(self):
        data = pd.read_csv(self.data_df, sep ="\t", header=None, names = ["user_id", "movie_id", "rating", "timestamp"])
        data = data.drop(['timestamp'], axis = 1)

        top_n_movies = data.groupby('movie_id').count().sort_values('user_id', ascending = False).head(self.n_movies).reset_index()['movie_id']
        self.movies = self.movies[self.movies.movie_id.isin(top_n_movies)].reset_index(drop=True)
        
        data = data[data.movie_id.isin(top_n_movies)]
        return data.sample(frac=1, random_state=self.seed)

    def _get_reward(self, action):
        user_id, _ = self._get_user_movie_id()
        movie_id = self._get_movie_id(action)
        watched_movies = self.data[self.data['user_id'] == user_id]['movie_id'].to_numpy()

        if movie_id in  watched_movies:
            user_rating = int(self.data[(self.data['user_id'] == user_id) & (self.data['movie_id'] == movie_id)]['rating'])
            # print(f'user {user_id} watched {movie_id} and give {user_rating} out of 5')
            if user_rating >= 3:
                reward = 1
            else:
                reward = 0
        else:
            # print(f'user {user_id} did not watch {movie_id}')
            reward = 0

        return reward

    def _get_user_movie_id(self):
        return self.data.iloc[self.local_step_counter]['user_id'], self.data.iloc[self.local_step_counter]['movie_id'] 
    
    def _get_movie_id(self, action):
        return self.movies['movie_id'].iloc[action]

if __name__ == '__main__':
   env = MovieRecoEnv(user_df = '../../../data/u.user', item_df = '../../../data/u.item', data_df = '../../../data/u.data')

