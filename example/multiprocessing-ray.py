import multiprocessing as mp
import time

from cpprb import ReplayBuffer, MPPrioritizedReplayBuffer
import numpy as np
import ray


class Model:
    def __init__(self):
        self.w = None

    def train(self, transitions):
        # Update Weight and return |TD|
        return absTD

    def __call__(self, obs):
        return act

@ray.remote
def explorer(global_rb, env_dict, q, stop):
    buffer_size = 100

    local_rb = ReplayBuffer(buffer_size, env_dict)
    env = gym.make("CartPole-v1")

    model = Model()

    obs = env.reset()
    while not stop.is_set():
        if not q.empty():
            w = q.get()
            model.w = w

        act = model(obs)
        next_obs, rew, done = env.step(act)
        local_rb.add(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)

        if done or local_rb.get_stored_size() == buffer_size():
            global_rb.add(**local_rb.get_all_transitions())
            local_rb.clear()
            obs = env.reset()
        else:
            obs = next_obs

    return None


def run():
    n_explorers = 4

    nwarmup = 100
    ntrain = int(1e+6)
    update_freq = 100


    buffer_size = 1e+6
    env_dict = {"obs": {},
                "act": {},
                "rew": {},
                "next_obs": {},
                "done": {}}
    alpha = 0.5
    beta = 0.4
    batch_size = 32


    ray.init()

    encoded = base64.b64encode(authkey)
    def auth_fn(*args):
        mp.current_process().authkey = base64.b64decode(encoded)
    ray.worker.global_worker.run_function_on_all_workers(auth_fn)

    m = mp.get_context().Manager()
    q = m.Queue()
    stop = m.Event()
    stop.clear()

    rb = MPPrioritizedReplayBuffer(buffer_size, env_dict, alpha=alpha,
                                   ctx=m, backend="SharedMemory")

    model = Model()

    explorers = []

    for _ in range(n_explorers):
        explorers.append(explorer.remote(rb, env_dict, q, stop))


    while rb.get_stored_size() < nwarmup:
        time.time(1)

    for i in range(ntrain):
        s = rb.sample(batch_size, beta)
        absTD = model.train(s)
        rb.update_priorities(s["indexes"], absTD)

        if i % update_freq == 0:
            q.put(model.w)

    stop.set()
    ray.get(explorers)

    m.shutdown()

if __name__ == "__main__":
    run()
