import os
import datetime

import numpy as np

import gym

import tensorflow as tf
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.summary import create_file_writer

from scipy.special import softmax

from cpprb import ReplayBuffer,PrioritizedReplayBuffer


gamma = 0.99
batch_size = 1024

N_iteration = int(1e+5)
N_show = 10
target_update_freq = 50

per_train = 100

prioritized = True

egreedy = True

# Log
dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join("logs", dir_name)
writer = create_file_writer(logdir + "/metrics")
writer.set_as_default()


# Env
env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env,
                           logdir + "/video/",
                           force=True,
                           video_callable=(lambda ep: ep % 50 == 0))

# For CartPole: input 4, output 2
model = Sequential([Dense(64,activation='relu',input_shape=(observation.shape)),
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
env_dict = {"obs":{"shape": observation.shape},
            "act":{"shape": 1,"dtype": np.ubyte},
            "rew": {},
            "next_obs": {"shape": observation.shape},
            "done": {}}

if prioritized:
    rb = PrioritizedReplayBuffer(buffer_size,env_dict)
else:
    rb = ReplayBuffer(buffer_size,env_dict)


@tf.function
def Q_func(model,obs,act,act_shape):
    return tf.reduce_sum(model(obs) * tf.one_hot(act,depth=act_shape), axis=1)

@tf.function
def DQN_target_func(target,next_obs,rew,done,gamma):
    return gamma*tf.reduce_max(target(next_obs),axis=1)*(1.0-done) + rew

@tf.function
def Double_DQN_target_func(model,target,next_obs,rew,done,gamma,act_shape):
    act = tf.math.argmax(model(next_obs),axis=1)
    return gamma*tf.reduce_sum(target(next_obs)*tf.one_hot(act,depth=act_shape), axis=1)*(1.0-done) + rew


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
        rb.on_episode_end()


sum_reward = 0
n_episode = 0
for n_step in range(N_iteration):
    Q = model(observation)

    if egreedy:
        if np.random.rand() < 0.9:
            action = tf.math.argmax(Q)
        else:
            action = env.action_space.sample()
    else:
        actions = softmax(Q)
        action = np.random.choice(actions.shape[0],p=actions)

    next_observation, reward, done, info = env.step(action)
    sum_reward += reward
    rb.add(obs=observation,
           act=action,
           rew=reward,
           next_obs=next_observation,
           done=done)
    observation = next_observation

    sample = rb.sample(batch_size)
    weights = sample["weights"].ravel() if prioritized else tf.constant(1.0)

    with tf.GradientTape as tape:
        tape.watch(model.trainable_weights)
        Q =  Q_func(model,
                    tf.constant(sample["obs"]),
                    tf.constant(sample["act"].ravel()),
                    tf.constant(env.action_space.n))
        target_Q = DQN_target_func(target_model,
                                   tf.constant(sample['next_obs']),
                                   tf.constant(sample["rew"].ravel()),
                                   tf.constant(sample["done"].ravel()),
                                   tf.constant(gamma))
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
        rb.on_episode_end()
        tf.summary.scalar("total reward",data=sum_reward,step=n_episode)
        sum_reward = 0
        n_episode += 1

    if n_step % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

    tf.summary.scalar("reward",data=reward,step=n_step)
