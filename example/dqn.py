import os
import datetime

import numpy as np

import gym
from scipy.special import softmax


import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.summary import create_file_writer


from cpprb import ReplayBuffer, PrioritizedReplayBuffer


gamma = 0.99
batch_size = 1024

N_iteration = int(1e+5)
target_update_freq = 50
eval_freq = 100

egreedy = 0.1



# Log
dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join("logs", dir_name)
writer = create_file_writer(logdir + "/metrics")
writer.set_as_default()


# Env
env = gym.make('CartPole-v1')
eval_env = gym.make('CartPole-v1')

# For CartPole: input 4, output 2
model = Sequential([Dense(64,activation='relu',
                          input_shape=(env.observation_space.shape)),
                    Dense(64,activation='relu'),
                    Dense(env.action_space.n)])
target_model = clone_model(model)


# Loss Function

@tf.function
def Huber_loss(absTD):
    return tf.where(absTD < 1.0, absTD, tf.math.square(absTD))

@tf.function
def MSE(absTD):
    return tf.math.square(absTD)

loss_func = Huber_loss


optimizer = Adam()


buffer_size = 1e+6
env_dict = {"obs":{"shape": env.observation_space.shape},
            "act":{"shape": 1,"dtype": np.ubyte},
            "rew": {},
            "next_obs": {"shape": env.observation_space.shape},
            "done": {}}

# Nstep
nstep = 3
# nstep = False

if nstep:
    Nstep = {"size": nstep, "rew": "rew", "next": "next_obs"}
    discount = tf.constant(gamma ** nstep)
else:
    Nstep = None
    discount = tf.constant(gamma)


# Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
# See https://ymd_h.gitlab.io/cpprb/features/per/
prioritized = True

if prioritized:
    rb = PrioritizedReplayBuffer(buffer_size,env_dict,Nstep=Nstep)

    # Beta linear annealing
    beta = 0.4
    beta_step = (1 - beta)/N_iteration
else:
    rb = ReplayBuffer(buffer_size,env_dict,Nstep=Nstep)


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


target_func = DQN_target_func



def evaluate(model,env):
    obs = env.reset()
    total_rew = 0

    while True:
        Q = tf.squeeze(model(obs.reshape(1,-1)))
        act = np.argmax(Q)
        obs, rew, done, _ = env.step(act)
        total_rew += rew

        if done:
            return total_rew

# Start Experiment

observation = env.reset()

# Warming up
for n_step in range(100):
    action = env.action_space.sample() # Random Action
    next_observation, reward, done, info = env.step(action)
    rb.add(obs=observation,
           act=action,
           rew=reward,
           next_obs=next_observation,
           done=done)
    observation = next_observation
    if done:
        env.reset()
        rb.on_episode_end()


n_episode = 0
observation = env.reset()
for n_step in range(N_iteration):
    Q = tf.squeeze(model(observation.reshape(1,-1)))

    if np.random.rand() < egreedy:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)

    next_observation, reward, done, info = env.step(action)
    rb.add(obs=observation,
           act=action,
           rew=reward,
           next_obs=next_observation,
           done=done)
    observation = next_observation

    if prioritized:
        sample = rb.sample(batch_size,beta)
        beta += beta_step
    else:
        sample = rb.sample(batch_size)

    weights = sample["weights"].ravel() if prioritized else tf.constant(1.0)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_weights)
        Q =  Q_func(model,
                    tf.constant(sample["obs"]),
                    tf.constant(sample["act"].ravel()),
                    tf.constant(env.action_space.n))
        target_Q = target_func(model,target_model,
                               tf.constant(sample['next_obs']),
                               tf.constant(sample["rew"].ravel()),
                               tf.constant(sample["done"].ravel()),
                               discount,
                               tf.constant(env.action_space.n))
        absTD = tf.math.abs(target_Q - Q)
        loss = tf.reduce_mean(loss_func(absTD)*weights)

    grad = tape.gradient(loss,model.trainable_weights)
    optimizer.apply_gradients(zip(grad,model.trainable_weights))


    if prioritized:
        Q =  Q_func(model,
                    tf.constant(sample["obs"]),
                    tf.constant(sample["act"].ravel()),
                    tf.constant(env.action_space.n))
        absTD = tf.math.abs(target_Q - Q)
        rb.update_priorities(sample["indexes"],absTD)

    if done:
        env.reset()
        rb.on_episode_end()
        n_episode += 1

    if n_step % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

    if n_step % eval_freq == eval_freq-1:
        eval_rew = evaluate(model,eval_env)
        tf.summary.scalar("episode reward vs training step",data=eval_rew,step=n_step)
