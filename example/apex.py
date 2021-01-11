from multiprocessing import Process, Event, SimpleQueue
import time

import gym
import numpy as np
from tqdm import tqdm

from cpprb import ReplayBuffer, MPPrioritizedReplayBuffer


class MyModel:
    def __init__(self):
        self._weights = 0

    def get_action(self,obs):
        # Implement action selection
        return 0

    def abs_TD_error(self,sample):
        # Implement absolute TD error
        return np.zeros(sample["obs"].shape[0])

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self,w):
        self._weights = w

    def train(self,sample):
        # Implement model update
        pass


def explorer(global_rb,env_dict,is_training_done,queue):
    local_buffer_size = int(1e+2)
    local_rb = ReplayBuffer(local_buffer_size,env_dict)

    model = MyModel()
    env = gym.make("CartPole-v1")

    obs = env.reset()
    while not is_training_done.is_set():
        if not queue.empty():
            w = queue.get()
            model.weights = w

        action = model.get_action(obs)
        next_obs, reward, done, _ = env.step(action)
        local_rb.add(obs=obs,act=action,rew=reward,next_obs=next_obs,done=done)

        if done:
            local_rb.on_episode_end()
            obs = env.reset()
        else:
            obs = next_obs

        if local_rb.get_stored_size() == local_buffer_size:
            local_sample = local_rb.get_all_transitions()
            local_rb.clear()

            absTD = model.abs_TD_error(local_sample)
            global_rb.add(**local_sample,priorities=absTD)


def learner(global_rb,queues):
    batch_size = 64
    n_warmup = 100
    n_training_step = int(1e+4)
    explorer_update_freq = 100

    model = MyModel()

    while global_rb.get_stored_size() < n_warmup:
        time.sleep(1)

    for step in tqdm(range(n_training_step)):
        sample = global_rb.sample(batch_size)

        model.train(sample)
        absTD = model.abs_TD_error(sample)
        global_rb.update_priorities(sample["indexes"],absTD)

        if step % explorer_update_freq == 0:
            w = model.weights
            for q in queues:
                q.put(w)


if __name__ == "__main__":
    buffer_size = int(1e+6)
    env_dict = {"obs": {"shape": 4},
                "act": {},
                "rew": {},
                "next_obs": {"shape": 4},
                "done": {}}
    n_explorer = 4

    global_rb = MPPrioritizedReplayBuffer(buffer_size,env_dict)

    is_training_done = Event()
    is_training_done.clear()

    qs = [SimpleQueue() for _ in range(n_explorer)]
    ps = [Process(target=explorer,
                  args=[global_rb,env_dict,is_training_done,q])
          for q in qs]

    for p in ps:
        p.start()

    learner(global_rb,qs)
    is_training_done.set()

    for p in ps:
        p.join()

    print(global_rb.get_stored_size())
