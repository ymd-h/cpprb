import numpy as np
import perfplot
import gc

# OpenAI/Baselines: https://github.com/openai/baselines
# Requires TensorFlow 1.14 instead of 2
from baselines.deepq.replay_buffer import (ReplayBuffer as bRB,
                                           PrioritizedReplayBuffer as bPRB)

# Ray/RLlib: https://github.com/ray-project/ray
# Requires Pandas, even though wich is not in `install_requires`
from ray.rllib.execution.replay_buffer import (ReplayBuffer as rRB,
                                               PrioritizedReplayBuffer as rPRB)
from ray.rllib.policy.sample_batch import SampleBatch

# Chainer/ChainerRL: https://github.com/chainer/chainerrl
from chainerrl.replay_buffers import (ReplayBuffer as cRB,
                                      PrioritizedReplayBuffer as cPRB)

from cpprb import (ReplayBuffer as RB,
                   PrioritizedReplayBuffer as PRB)


# Configulation
buffer_size = 2**12

obs_shape = 15
act_shape = 3

alpha = 0.4
beta  = 0.4

env_dict = {"obs": {"shape": obs_shape},
            "act": {"shape": act_shape},
            "next_obs": {"shape": obs_shape},
            "rew": {},
            "done": {}}


# Initialize Replay Buffer
brb = bRB(buffer_size)
rrb = rRB(buffer_size)
crb = cRB(buffer_size)
rb  =  RB(buffer_size,env_dict)


# Initialize Prioritized Replay Buffer
bprb = bPRB(buffer_size,alpha=alpha)
rprb = rPRB(buffer_size,alpha=alpha)
cprb = cPRB(buffer_size,alpha=alpha,beta0=beta,betasteps=None)
prb  =  PRB(buffer_size,env_dict,alpha=alpha)



# Helper Function
def env(n):
    e = {"obs": np.ones((n,obs_shape)),
         "act": np.zeros((n,act_shape)),
         "next_obs": np.ones((n,obs_shape)),
         "rew": np.zeros(n),
         "done": np.zeros(n)}
    return e

def add_b(_rb):
    """ Add for Baselines
    """
    def add(e):
        for i in range(e["obs"].shape[0]):
            _rb.add(obs_t=e["obs"][i],
                    action=e["act"][i],
                    reward=e["rew"][i],
                    obs_tp1=e["next_obs"][i],
                    done=e["done"][i])
    return add

def add_r(_rb):
    """ Add for RLlib

    Notes
    -----
    Even `ReplayBuffer` requires `weight` parameter (but don't use it).
    """
    def add(e):
        for i in range(e["obs"].shape[0]):
            _rb.add(SampleBatch(obs_t=[e["obs"][i]]),
                                action=[e["act"][i]],
                                reward=[e["rew"][i]],
                                obs_tp1=[e["next_obs"][i]],
                                done=[e["done"][i]],
                    weight=0.5)
    return add

def add_c(_rb):
    """ Add for ChainerRL
    """
    def add(e):
        for i in range(e["obs"].shape[0]):
            _rb.append(state=e["obs"][i],
                       action=e["act"][i],
                       reward=e["rew"][i],
                       next_state=e["next_obs"][i],
                       is_state_terminal=e["done"][i])
    return add

def sample_c(_rb):
    """ Force sample from ChainerRL PrioritizedReplayBuffer
    """
    def sample(n):
        _rb.memory.wait_priority_after_sampling = False
        return _rb.sample(n)

    return sample


# ReplayBuffer.add
perfplot.save(filename="ReplayBuffer_add.png",
              setup = env,
              time_unit="ms",
              kernels = [add_b(brb),
                         add_r(rrb),
                         add_c(crb),
                         lambda e: rb.add(**e)],
              labels = ["OpenAI/Baselines","Ray/RLlib","Chainer/ChainerRL","cpprb"],
              n_range = [n for n in range(1,102,10)],
              xlabel = "Step size added at once",
              title = "Replay Buffer Add Speed",
              logx = False,
              logy = False,
              equality_check = None)


# Fill Buffers
for _ in range(buffer_size):
    o = np.random.rand(obs_shape) # [0,1)
    a = np.random.rand(act_shape)
    r = np.random.rand(1)
    d = np.random.randint(2) # [0,2) == 0 or 1
    brb.add(obs_t=o,action=a,reward=r,obs_tp1=o,done=d)
    rrb.add(obs_t=o,action=a,reward=r,obs_tp1=o,done=d,weight=0.5)
    crb.append(state=o,action=a,reward=r,next_state=o,is_state_terminal=d)
    rb.add(obs=o,act=a,rew=r,next_obs=o,done=d)


# ReplayBuffer.sample
perfplot.save(filename="ReplayBuffer_sample.png",
              setup = lambda n: n,
              time_unit="ms",
              kernels = [brb.sample,
                         rrb.sample,
                         crb.sample,
                         rb.sample],
              labels = ["OpenAI/Baselines",
                        "Ray/RLlib",
                        "Chainer/ChainerRL",
                        "cpprb"],
              n_range = [2**n for n in range(1,8)],
              xlabel = "Batch size",
              title = "Replay Buffer Sample Speed",
              logx = False,
              logy = False,
              equality_check=None)


# PrioritizedReplayBuffer.add
perfplot.save(filename="PrioritizedReplayBuffer_add.png",
              time_unit="ms",
              setup = env,
              kernels = [add_b(bprb),
                         add_r(rprb),
                         add_c(cprb),
                         lambda e: prb.add(**e)],
              labels = ["OpenAI/Baselines",
                        "Ray/RLlib",
                        "Chainer/ChainerRL",
                        "cpprb"],
              n_range = [n for n in range(1,102,10)],
              xlabel = "Step size added at once",
              title = "Prioritized Replay Buffer Add Speed",
              logx = False,
              logy = False,
              equality_check=None)


# Fill Buffers
for _ in range(buffer_size):
    o = np.random.rand(obs_shape) # [0,1)
    a = np.random.rand(act_shape)
    r = np.random.rand(1)
    d = np.random.randint(2) # [0,2) == 0 or 1
    p = np.random.rand(1)

    # OpenAI/Baselines cannot set priority together.
    idx = bprb._next_idx
    bprb.add(obs_t=o,action=a,reward=r,obs_tp1=o,done=d)
    bprb.update_priorities([idx],[p])

    rprb.add(obs_t=o,action=a,reward=r,obs_tp1=o,done=d,weight=p)

    # Directly access internal PrioritizedBuffer,
    # since ChainerRL/PrioritizedReplayBuffer has no API to set priority.
    cprb.memory.append([{"state":o,
                        "action":a,
                        "reward":r,
                        "next_state":o,
                        "is_state_terminal":d}],
                       priority=p)

    prb.add(obs=o,act=a,rew=r,next_obs=o,done=d,priority=p)


perfplot.save(filename="PrioritizedReplayBuffer_sample.png",
              time_unit="ms",
              setup = lambda n: n,
              kernels = [lambda n: bprb.sample(n,beta=beta),
                         lambda n: rprb.sample(n,beta=beta),
                         sample_c(cprb),
                         lambda n: prb.sample(n,beta=beta)],
              labels = ["OpenAI/Baselines",
                        "Ray/RLlib",
                        "Chainer/ChainerRL",
                        "cpprb"],
              n_range = [2**n for n in range(1,9)],
              xlabel = "Batch size",
              title = "Prioritized Replay Buffer Sample Speed",
              logx=False,
              logy=False,
              equality_check=None)
