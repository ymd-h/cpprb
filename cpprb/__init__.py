"""
cpprb: Fast Flexible Replay Buffer Library

cpprb provides replay buffer classes for reinforcement learning.
Details are described at `Home Page <https://ymd_h.gitlab.io/cpprb/>`_.

Examples
--------
Replay Buffer classes can be imported from ``cpprb`` package.

>>> from cpprb import ReplayBuffer

These buffer classes can be created by specifying ``env_dict``.

>>> buffer_size = 1e6
>>> env_dict = {"obs": {}, "act": {}, "rew": {}, "next_obs": {}, "done": {}}
>>> rb = ReplayBuffer(buffer_size, env_dict)

When adding transitions, all values must be passed as keyword arguments.

>>> rb.add(obs=1, act=1, rew=0.5, next_obs=2, done=0)

You can also add multiple transitions simultaneously.

>>> rb.add(obs=[1, 2], act=[1, 2], rew=[0.5, 0.3], next_obs=[2, 3], done=[0, 1])

At the episode end, users must call ``on_episode_end()`` method.

>>> rb.on_episode_end()

Transitions can be sampled according to these buffer's algorithms (e.g. random).

>>> sample = rb.sample(32)
"""

import contextlib

__all__ = [
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "MPReplayBuffer",
    "MPPrioritizedReplayBuffer",
    "SelectiveReplayBuffer",
    "ReverseReplayBuffer",
    "LaBERmean",
    "LaBERlazy",
    "LaBERmax",
    "HindsightReplayBuffer",
    "create_buffer",
    "train",
]


from cpprb.PyReplayBuffer import (  # noqa: I001
    MPPrioritizedReplayBuffer,
    MPReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    ReverseReplayBuffer,
    SelectiveReplayBuffer,
    create_buffer,
    train,
)
from cpprb.HER import HindsightReplayBuffer
from cpprb.LaBER import LaBERlazy, LaBERmax, LaBERmean

with contextlib.suppress(ImportError):
    # If gym is not installed, util functions are not defined.
    from cpprb.util import create_before_add_func, create_env_dict  # noqa: F401
