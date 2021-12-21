from gym.envs.registration import register

register(
    id = 'RecoEnv-v0',
    entry_point = 'movie_env.envs:MovieRecoEnv',
    kwargs={'user_df': './data/u.user', 'item_df': './data/u.item', 'data_df': './data/u.data'}
)