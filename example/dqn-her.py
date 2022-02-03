import os
import datetime

import numpy as np

import gym
from gym.spaces import Box, Discrete

import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.summary import create_file_writer


from cpprb import HindsightReplayBuffer


class BitFlippingEnv(gym.Env):
    """
    bit-flipping environment: https://arxiv.org/abs/1707.01495

    * Environment has n-bit state.
    * Initial state and goal state are randomly selected.
    * Action is one of the 0, ..., n-1, which flips single bit
    * Reward is 0 if state == goal, otherwise reward is -1. (Sparse Binary Reward)

    Simple RL algorithms tend to fail for large ``n`` like ``n > 40``
    """
    def __init__(self, n):
        self.np_random = np.random.default_rng()
        seeds = self.np_random.spawn(2)
        self.observation_space = Box(low=0, high=1, shape=(n,), dtype=int,
                                     seed=np.random.SeedSequence(seeds[0]))
        self.action_space = Discrete(self.n, seed=seeds[1])

    def step(self, action):
        action = int(action)
        self.bit[action] = 1 - self.bit[action]
        done = (self.bit == self.goal).all()
        rew = 0 if done else -1
        return self.bit.copy(), rew, done, {}

    def reset(self):
        self.bit = self.np_random.integers(low=0, high=1, size=self.action_space.n,
                                           endpoint=True, dtype=int)
        self.goal = self.np_random.integers(low=0, high=1, size=self.action_space.n,
                                            endpoint=True, dtype=int)
        return self.bit.copy()



gamma = 0.99
batch_size = 64

N_iteration = int(1e+5)
nwarmup = 100

target_update_freq = 1000
eval_freq = 100

egreedy = 0.1

max_episode_len = 100

nbit = 40


# Log
dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join("logs", dir_name)
writer = create_file_writer(logdir + "/metrics")
writer.set_as_default()


# Env
env = BitFlippingEnv(nbit)
eval_env = BitFlippingEnv(nbit)


model = Sequential([Dense(64,activation='relu',
                          input_shape=(env.observation_space.shape[0] * 2,)),
                    Dense(64,activation='relu'),
                    Dense(env.action_space.n)])
target_model = clone_model(model)


# Loss Function

@tf.function
def Huber_loss(absTD):
    return tf.where(absTD > 1.0, absTD, tf.math.square(absTD))

@tf.function
def MSE(absTD):
    return tf.math.square(absTD)

loss_func = Huber_loss


optimizer = Adam()


buffer_size = 1e+6
env_dict = {"obs":{"shape": env.observation_space.shape},
            "act":{"shape": 1,"dtype": np.ubyte},
            "next_obs": {"shape": env.observation_space.shape}}


discount = tf.constant(gamma)


# Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
# See https://ymd_h.gitlab.io/cpprb/features/per/
prioritized = True


# Hindsigh Experience Replay : https://arxiv.org/abs/1707.01495
# See https://ymd_h.gitlab.io/cpprb/features/her/
rb = HindsightReplayBuffer(buffer_size, env_dict,
                           max_episode_len = max_episode_len,
                           rew_func = lambda x,a,g: -1*(s!=g),
                           prioritized = prioritized)

if prioritized:
    # Beta linear annealing
    beta = 0.4
    beta_step = (1 - beta)/N_iteration


def sg(state, goal):
    state = state.reshape((state.shape[0], 1))
    goal = goal.reshape((goal.shape[0], 1))
    return tf.constant(np.concatenate((state, goal), axis=1), dtype=tf.float32)

@tf.function
def Q_func(model,obs,act,act_shape):
    return tf.reduce_sum(model(obs) * tf.one_hot(act,depth=act_shape), axis=1)

@tf.function
def DQN_target_func(model,target,next_obs,rew,done,gamma,act_shape):
    return gamma*tf.reduce_max(target(next_obs),axis=1)*(1.0-done) + rew

@tf.function
def Double_DQN_target_func(model,target,next_obs,rew,done,gamma,act_shape):
    """
    Double DQN: https://arxiv.org/abs/1509.06461
    """
    act = tf.math.argmax(model(next_obs),axis=1)
    return gamma*tf.reduce_sum(target(next_obs)*tf.one_hot(act,depth=act_shape), axis=1)*(1.0-done) + rew


target_func = Double_DQN_target_func



def evaluate(model,env):
    obs = env.reset()
    goal = env.goal.copy().reshape((1, -1))

    n_episode = 20
    i_episode = 0

    success = 0
    ep = 0
    while i_episode < n_episode:
        Q = tf.squeeze(model(sg(obs.reshape((1, -1), goal))))
        act = np.argmax(Q)
        obs, _, done, _ = env.step(act)
        ep += 1

        if done or (ep >= max_episode_len):
            if done:
                success += 1
            obs = env.reset()
            goal = env.goal.copy().reshape((1, -1))

            i_episode += 1
            ep = 0

    return success / n_episode


# Start Experiment

n_episode = 0
obs = env.reset()
goal = env.goal.copy().reshape((1, -1))
ep = 0

for n_step in range(N_iteration):
    if np.random.rand() < egreedy:
        act = env.action_space.sample()
    else:
        Q = tf.squeeze(model(sg(obs.reshape(1, -1), goal)))
        act = np.argmax(Q)

    next_obs, rew, done, info = env.step(act)
    ep += 1

    rb.add(obs=obs,
           act=act,
           rew=rew,
           next_obs=next_obs,
           done=done)

    if done or (ep >= max_episode_len):
        obs = env.reset()
        goal = env.goal.copy().reshape((1, -1))
        rb.on_episode_end()
        n_episode += 1
        ep = 0
    else:
        obs = next_obs

    if rb.get_stored_size() < nwarmup:
        continue

    if prioritized:
        sample = rb.sample(batch_size, beta)
        beta += beta_step
    else:
        sample = rb.sample(batch_size)

    weights = sample["weights"].ravel() if prioritized else tf.constant(1.0)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_weights)
        Q =  Q_func(model,
                    sg(sample["obs"], sample["goal"]),
                    tf.constant(sample["act"].ravel()),
                    tf.constant(env.action_space.n))
        target_Q = tf.stop_gradient(target_func(model,target_model,
                                                sg(sample["next_obs"],sample["goal"]),
                                                tf.constant(sample["rew"].ravel()),
                                                tf.constant(sample["done"].ravel()),
                                                discount,
                                                tf.constant(env.action_space.n)))
        absTD = tf.math.abs(target_Q - Q)
        loss = tf.reduce_mean(loss_func(absTD)*weights)

    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    tf.summary.scalar("Loss vs training step", data=loss, step=n_step)


    if prioritized:
        Q =  Q_func(model,
                    sg(sample["obs"], sample["goal"]),
                    tf.constant(sample["act"].ravel()),
                    tf.constant(env.action_space.n))
        absTD = tf.math.abs(target_Q - Q)
        rb.update_priorities(sample["indexes"], absTD)


    if n_step % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

    if n_step % eval_freq == eval_freq-1:
        eval_rew = evaluate(model, eval_env)
        tf.summary.scalar("success rate vs training step",
                          data=eval_rew, step=n_step)
