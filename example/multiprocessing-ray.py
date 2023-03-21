# See: https://ymd_h.gitlab.io/cpprb/examples/mp_with_ray/

import base64
import multiprocessing as mp
import time

from cpprb import ReplayBuffer, MPPrioritizedReplayBuffer
import gym
import numpy as np
import ray


class Model:
    def __init__(self, env):
        self.env = env
        self.w = None

    def train(self, transitions):
        """
        Update model weights and return |TD|
        """
        absTD = np.zeros(shape=(transitions["obs"].shape[0],))
        # omit
        return absTD

    def __call__(self, obs):
        """
        Choose action from observation
        """
        # omit
        act = self.env.action_space.sample()
        return act

@ray.remote
class Explorer:
    encoded = base64.b64encode(mp.current_process().authkey)

    def __init__(self):
        mp.current_process().authkey = base64.b64decode(self.encoded)

    def run(self, env_name, global_rb, env_dict, q, stop):
        try:
            buffer_size = 200

            local_rb = ReplayBuffer(buffer_size, env_dict)
            env = gym.make(env_name)
            model = Model(env)

            reset = env.reset()
            if not isinstance(reset, tuple):
                # Gym Old API
                obs = reset
            else:
                # Gym New API
                obs, _ = reset

            while True:
                if stop.is_set():
                    print("Stop")
                    break
                if not q.empty():
                    w = q.get()
                    model.w = w

                act = model(obs)

                stepped = env.step(act)
                if len(stepped) == 4:
                    # Gym Old API
                    next_obs, rew, done, _ = stepped
                else:
                    # Gym New API
                    next_obs, rew, term, trunc, _ = stepped
                    done = term | trunc

                local_rb.add(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)

                if done or local_rb.get_stored_size() == buffer_size:
                    local_rb.on_episode_end()
                    global_rb.add(**local_rb.get_all_transitions())
                    local_rb.clear()
                    reset = env.reset()
                    if not isinstance(reset, tuple):
                        # Gym Old API
                        obs = reset
                    else:
                        # Gym New API
                        obs, _ = reset
                else:
                    obs = next_obs
        finally:
            stop.set()
        return None


def run():
    n_explorers = 4

    nwarmup = 100
    ntrain = int(1e+2)
    update_freq = 100
    env_name = "CartPole-v1"

    env = gym.make(env_name)

    buffer_size = 1e+6
    env_dict = {"obs": {"shape": env.observation_space.shape},
                "act": {},
                "rew": {},
                "next_obs": {"shape": env.observation_space.shape},
                "done": {}}
    alpha = 0.5
    beta = 0.4
    batch_size = 32


    ray.init()


    # `BaseContext.Manager()` automatically starts `SyncManager`
    # Ref: https://github.com/python/cpython/blob/3.9/Lib/multiprocessing/context.py#L49-L58
    m = mp.get_context().Manager()
    q = [m.Queue() for _ in range(n_explorers)]
    stop = m.Event()
    stop.clear()

    rb = MPPrioritizedReplayBuffer(buffer_size, env_dict, alpha=alpha,
                                   ctx=m, backend="SharedMemory")

    model = Model(env)

    explorers = []
    jobs = []

    print("Start Explorers")
    for i in range(n_explorers):
        explorers.append(Explorer.remote())
        jobs.append(explorers[-1].run.remote(env_name, rb, env_dict, q[i], stop))


    print("Start Warmup")
    while rb.get_stored_size() < nwarmup and not stop.is_set():
        time.sleep(1)

    print("Start Training")
    for i in range(ntrain):
        if stop.is_set():
            break

        s = rb.sample(batch_size, beta)
        absTD = model.train(s)
        rb.update_priorities(s["indexes"], absTD)

        if i % update_freq == 0:
            q[i].put(model.w)

    print("Finish Training")

    stop.set()
    _, still_running = ray.wait(jobs, timeout=10)

    m.shutdown()

if __name__ == "__main__":
    run()
