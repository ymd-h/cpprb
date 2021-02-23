import os
import datetime
import io
import base64

import numpy as np

import gym

import tensorflow as tf
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import InputLayer,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.summary import create_file_writer

from scipy.special import softmax

from cpprb import ReplayBuffer,PrioritizedReplayBuffer
import cpprb.gym


gamma = 0.99
batch_size = 1024

N_iteration = 101
N_show = 10

per_train = 100

prioritized = True

egreedy = True

loss = "huber_loss"
# loss = "mean_squared_error"

dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


logdir = os.path.join("logs", dir_name)
writer = create_file_writer(logdir + "/metrics")
writer.set_as_default()

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env,
                           logdir + "/video/",
                           force=True,
                           video_callable=(lambda ep: ep % 50 == 0))

model = Sequential([InputLayer(input_shape=(observation.shape)), # 4 for CartPole
                    Dense(64,activation='relu'),
                    Dense(64,activation='relu'),
                    Dense(env.action_space.n)]) # 2 for CartPole

target_model = clone_model(model)


optimizer = Adam()
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)


model.compile(loss =  loss,
              optimizer = optimizer,
              metrics=['accuracy'])


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

action_index = np.arange(env.action_space.n).reshape(1,-1)



def Q_func(model,obs,act,act_shape):
    return tf.reduce_sum(model(obs) * tf.one_hot(act,depth=act_shape), axis=1)


def DQN_target_func(model,next_obs,rew,done,gamma):
    return gamma*tf.reduce_max(model(next_obs),axis=1)*(1.0-done) + rew


def Double_DQN_target_func(model,target,next_obs,rew,done,gamma,act_shape):
    act = tf.math.argmax(model(next_obs),axis=1)
    return gamma*tf.reduce_sum(target(next_obs)*tf.one_hot(act,depth=act_shape), axis=1)*(1.0-done) + rew


observation = env.reset()

# Warming up
for n_step in range(1000):
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
for n_step in range(N_iteration):
    actions = softmax(np.ravel(model.predict(observation.reshape(1,-1),
                                                 batch_size=1)))
    actions = actions / actions.sum()

    if egreedy:
        if np.random.rand() < 0.9:
            action = np.argmax(actions)
        else:
            action = env.action_space.sample()
    else:
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
    Q_pred = model.predict(sample["obs"])
    Q_true = target_model.predict(sample['next_obs']).max(axis=1,keepdims=True)*gamma*(1.0 - sample["done"]) + sample['rew']
    target = tf.where(tf.one_hot(tf.cast(tf.reshape(sample["act"],[-1]),
                                         dtype=tf.int32),
                                 env.action_space.n,
                                 True,False),
                      tf.broadcast_to(Q_true,[batch_size,env.action_space.n]),
                      Q_pred)

    if prioritized:
        TD = tf.reduce_mean(tf.math.abs(target - Q_pred))
        rb.update_priorities(sample["indexes"],TD)

    model.fit(x=sample['obs'],
              y=target,
              batch_size=batch_size,
              verbose = 0)

    if done:
        rb.on_episode_end()
        sum_reward = 0

    if n_step % 10 == 0:
        target_model.set_weights(model.get_weights())

    tf.summary.scalar("reward",data=sum_reward,step=n_episode)
