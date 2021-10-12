# Example for Large Batch Experience Replay (LaBER)
# Ref: https://arxiv.org/abs/2110.01528

import os
import datetime

import numpy as np

import gym

import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.summary import create_file_writer


from cpprb import ReplayBuffer, LaBERmean


gamma = 0.99
batch_size = 64

N_iteration = int(1e+5)
target_update_freq = 1000
eval_freq = 100

egreedy = 0.1


# Use 4 times larger batch for initial uniform sampling
# Use LaBER-mean, which is the best variant
m = 4
LaBER = LaBERmean(batch_size, m)


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
    return tf.where(absTD > 1.0, absTD, tf.math.square(absTD))

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


target_func = Double_DQN_target_func



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

    if np.random.rand() < egreedy:
        action = env.action_space.sample()
    else:
        Q = tf.squeeze(model(observation.reshape(1,-1)))
        action = np.argmax(Q)

    next_observation, reward, done, info = env.step(action)
    rb.add(obs=observation,
           act=action,
           rew=reward,
           next_obs=next_observation,
           done=done)
    observation = next_observation

    # Uniform sampling
    sample = rb.sample(batch_size * m)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_weights)
        Q =  Q_func(model,
                    tf.constant(sample["obs"]),
                    tf.constant(sample["act"].ravel()),
                    tf.constant(env.action_space.n))
        target_Q = tf.stop_gradient(target_func(model,target_model,
                                                tf.constant(sample['next_obs']),
                                                tf.constant(sample["rew"].ravel()),
                                                tf.constant(sample["done"].ravel()),
                                                discount,
                                                tf.constant(env.action_space.n)))
        absTD = tf.math.abs(target_Q - Q)

        # Sub-sample according to surrogate priorities
        #   When loss is L2 or Huber, and no activation at the last layer,
        #   |TD| is surrogate priority.
        sample = LaBER(priorities=absTD)
        indexes = tf.constant(sample["indexes"])
        weights = tf.constant(sample["weights"])

        absTD = tf.gather(absTD, indexes)
        assert absTD.shape == weights.shape, f"BUG: absTD.shape: {absTD.shae}, weights.shape {weights.shape}"

        loss = tf.reduce_mean(loss_func(absTD)*weights)

    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    tf.summary.scalar("Loss vs training step", data=loss, step=n_step)


    if done:
        env.reset()
        rb.on_episode_end()
        n_episode += 1

    if n_step % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

    if n_step % eval_freq == eval_freq-1:
        eval_rew = evaluate(model,eval_env)
        tf.summary.scalar("episode reward vs training step",data=eval_rew,step=n_step)
