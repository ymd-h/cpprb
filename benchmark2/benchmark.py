import gc
import itertools


import numpy as np
import perfplot
import tensorflow as tf

# DeepMind/Reverb: https://github.com/deepmind/reverb
import reverb

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
rb  =  RB(buffer_size,env_dict)


# Initialize Prioritized Replay Buffer
prb  =  PRB(buffer_size,env_dict,alpha=alpha)


# Initalize Reverb Server
server = reverb.Server(tables =[
    reverb.Table(name='ReplayBuffer',
                 sampler=reverb.selectors.Uniform(),
                 remover=reverb.selectors.Fifo(),
                 max_size=buffer_size,
                 rate_limiter=reverb.rate_limiters.MinSize(1)),
    reverb.Table(name='PrioritizedReplayBuffer',
                 sampler=reverb.selectors.Prioritized(alpha),
                 remover=reverb.selectors.Fifo(),
                 max_size=buffer_size,
                 rate_limiter=reverb.rate_limiters.MinSize(1))
])

client = reverb.Client(f"localhost:{server.port}")
tf_client = reverb.TFClient(f"localhost:{server.port}")


# Helper Function
def env(n):
    e = {"obs": np.ones((n,obs_shape)),
         "act": np.zeros((n,act_shape)),
         "next_obs": np.ones((n,obs_shape)),
         "rew": np.zeros(n),
         "done": np.zeros(n)}
    return e

def add_client(_rb,table):
    """ Add for Reverb Client
    """
    def add(e):
        n = e["obs"].shape[0]
        with _rb.writer(max_sequence_length=1) as _w:
            for i in range(n):
                _w.append([e["obs"][i],
                           e["act"][i],
                           e["rew"][i],
                           e["next_obs"][i],
                           e["done"][i]])
                _w.create_item(table,1,1.0)
    return add

def add_client_insert(_rb,table):
    """ Add for Reverb Client
    """
    def add(e):
        n = e["obs"].shape[0]
        for i in range(n):
            _rb.insert([e["obs"][i],
                        e["act"][i],
                        e["rew"][i],
                        e["next_obs"][i],
                        e["done"][i]],priorities={table: 1.0})
    return add

def add_tf_client(_rb,table):
    """ Add for Reverb TFClient
    """
    def add(e):
        n = e["obs"].shape[0]
        for i in range(n):
            _rb.insert([tf.constant(e["obs"][i]),
                        tf.constant(e["act"][i]),
                        tf.constant(e["rew"][i]),
                        tf.constant(e["next_obs"][i]),
                        tf.constant(e["done"])],
                       tf.constant([table]),
                       tf.constant([1.0],dtype=tf.float64))
    return add

def sample_client(_rb,table):
    """ Sample from Reverb Client
    """
    def sample(n):
        return [i for i in _rb.sample(table,num_samples=n)]

    return sample

def sample_tf_client(_rb,table):
    """ Sample from Reverb TFClient
    """
    def sample(n):
        return [_rb.sample(table,
                           [tf.float64,tf.float64,tf.float64,tf.float64,tf.float64])
                for _ in range(n)]

    return sample

def sample_tf_client_dataset(_rb,table):
    """ Sample from Reverb TFClient using dataset
    """
    def sample(n):
        dataset=_rb.dataset(table,
                            [tf.float64,tf.float64,tf.float64,tf.float64,tf.float64],
                            [4,1,1,4,1])
        return itertools.islice(dataset,n)
    return sample


# ReplayBuffer.add
perfplot.save(filename="ReplayBuffer_add2.png",
              setup = env,
              time_unit="ms",
              kernels = [add_client_insert(client,"ReplayBuffer"),
                         add_client(client,"ReplayBuffer"),
                         add_tf_client(tf_client,"ReplayBuffer"),
                         lambda e: rb.add(**e)],
              labels = ["DeepMind/Reverb: Client.insert",
                        "DeepMind/Reverb: Client.writer",
                        "DeepMind/Reverb: TFClient.insert",
                        "cpprb"],
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
    client.insert([o,a,r,o,d],priorities={"ReplayBuffer": 1.0})
    rb.add(obs=o,act=a,rew=r,next_obs=o,done=d)


# ReplayBuffer.sample
perfplot.save(filename="ReplayBuffer_sample2.png",
              setup = lambda n: n,
              time_unit="ms",
              kernels = [sample_client(client,"ReplayBuffer"),
                         sample_tf_client(tf_client,"ReplayBuffer"),
                         sample_tf_client_dataset(tf_client,"ReplayBuffer"),
                         rb.sample],
              labels = ["DeepMind/Reverb: Client.sample",
                        "DeepMind/Reverb: TFClient.sample",
                        "DeepMind/Reverb: TFClient.dataset",
                        "cpprb"],
              n_range = [2**n for n in range(1,8)],
              xlabel = "Batch size",
              title = "Replay Buffer Sample Speed",
              logx = False,
              logy = False,
              equality_check=None)


# PrioritizedReplayBuffer.add
perfplot.save(filename="PrioritizedReplayBuffer_add2.png",
              time_unit="ms",
              setup = env,
              kernels = [add_client_insert(client,"PrioritizedReplayBuffer"),
                         add_client(client,"PrioritizedReplayBuffer"),
                         add_tf_client(tf_client,"PrioritizedReplayBuffer"),
                         lambda e: prb.add(**e)],
              labels = ["DeepMind/Reverb: Client.insert",
                        "DeepMind/Reverb: Client.writer",
                        "DeepMind/Reverb: TFClient.insert",
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

    client.insert([o,a,r,o,d],priorities={"PrioritizedReplayBuffer": p})

    prb.add(obs=o,act=a,rew=r,next_obs=o,done=d,priority=p)


perfplot.save(filename="PrioritizedReplayBuffer_sample2.png",
              time_unit="ms",
              setup = lambda n: n,
              kernels = [sample_client(client,"PrioritizedReplayBuffer"),
                         sample_tf_client(tf_client,"PrioritizedReplayBuffer"),
                         sample_tf_client_dataset(tf_client,"PrioritizedReplayBuffer"),
                         lambda n: prb.sample(n,beta=beta)],
              labels = ["DeepMind/Reverb: Client.sample",
                        "DeepMind/Reverb: TFClient.sample",
                        "DeepMind/Reverb: TFClient.dataset",
                        "cpprb"],
              n_range = [2**n for n in range(1,9)],
              xlabel = "Batch size",
              title = "Prioritized Replay Buffer Sample Speed",
              logx=False,
              logy=False,
              equality_check=None)
