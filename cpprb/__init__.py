"""
cpprb: Fast Flexible Replay Buffer Library

cpprb provides replay buffer classes for reinforcement learning.
Details are described at `Home Page <https://ymd_h.gitlab.io/cpprb/>`_.

Examples
--------
Replay Buffer classes can be imported from ``cpprb`` package.

>>> from cpprb import ReplayBuffer

These buffer classes can be created by specifying ``env_dict``.

>>> buffer_size = 1e+6
>>> env_dict = {"obs": {}, "act": {}, "rew": {}, "next_obs": {}, "done": {}}
>>> rb = ReplayBuffer(buffer_size, env_dict)

When adding transitions, all values must be passed as keyword arguments.

>>> rb.add(obs=1, act=1, rew=0.5, next_obs=2, done=0)

You can also add multiple transitions simultaneously.

>>> rb.add(obs=[1,2], act=[1,2], rew=[0.5,0.3], next_obs=[2,3], done=[0,1])

At the episode end, users must call ``on_episode_end()`` method.

>>> rb.on_episode_end()

Transitions can be sampled according to these buffer's algorithms (e.g. random).

>>> sample = rb.sample(32)
"""

from .PyReplayBuffer import (ReplayBuffer,PrioritizedReplayBuffer,
                             MPReplayBuffer, MPPrioritizedReplayBuffer,
                             SelectiveReplayBuffer, ReverseReplayBuffer)

from .LaBER import LaBERmean, LaBERlazy, LaBERmax
from .HER import HindsightReplayBuffer

from .PyReplayBuffer import create_buffer, train

try:
    from .util import create_env_dict, create_before_add_func
except ImportError:
    # If gym is not installed, util functions are not defined.
    pass
