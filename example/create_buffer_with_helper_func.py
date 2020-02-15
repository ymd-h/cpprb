# create_buffer_with_helper_func.py
#
# Create `ReplayBuffer` for non simple space `gym.Env` with helper functions.


import gym
from cpprb import ReplayBuffer, create_env_dict, create_before_add_func

env = gym.make("Blackjack-v0")
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
# BlackjackEnv
#   observation_space: Tuple(Discrete(32),Discrete(11),Discrete(2))
#   action_space     : Discrete(2)


env_dict = create_env_dict(env)
# >>> env_dict
#{'act': {'add_shape': array([-1,  1]), 'dtype': numpy.int32, 'shape': 1},
# 'done': {'add_shape': array([-1,  1]), 'dtype': numpy.float32, 'shape': 1},
# 'next_obs0': {'add_shape': array([-1,  1]), 'dtype': numpy.int32, 'shape': 1},
# 'next_obs1': {'add_shape': array([-1,  1]), 'dtype': numpy.int32, 'shape': 1},
# 'next_obs2': {'add_shape': array([-1,  1]), 'dtype': numpy.int32, 'shape': 1},
# 'obs0': {'add_shape': array([-1,  1]), 'dtype': numpy.int32, 'shape': 1},
# 'obs1': {'add_shape': array([-1,  1]), 'dtype': numpy.int32, 'shape': 1},
# 'obs2': {'add_shape': array([-1,  1]), 'dtype': numpy.int32, 'shape': 1},
# 'rew': {'add_shape': array([-1,  1]), 'dtype': numpy.float32, 'shape': 1}}



rb = ReplayBuffer(256, env_dict)


obs = env.reset()
before_add = create_before_add_func(env)

for i in range(400):
    act = env.action_space.sample()
    next_obs, rew, done, _ = env.step(act)

    rb.add(**before_add(obs=obs,act=act,next_obs=next_obs,rew=rew,done=done))
    # Create `dict` for `ReplayBuffer.add`

    if done:
        obs = env.reset()
    else:
        obs = next_obs

