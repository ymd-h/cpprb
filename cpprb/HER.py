from typing import Dict, Callable, Optional, Iterable

import numpy as np

from cpprb import ReplayBuffer, PrioritizedReplayBuffer


class HindsightReplayBuffer:
    """
    Replay Buffer class for Hindsight Experience Replay (HER)

    Notes
    -----
    In Hindsight Experience Replay [1]_, failed transitions are considered
    as success transitions by re-labelling goal.

    References
    ----------
    .. [1] M. Andrychowicz et al, "Hindsight Experience Replay",
       Advances in Neural Information Processing Systems 30 (NIPS 2017),
       https://papers.nips.cc/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html
       https://arxiv.org/abs/1707.01495
    """
    def __init__(self,
                 size: int,
                 env_dict: Dict,
                 max_episode_len: int,
                 reward_func: Callable, *,
                 goal_func: Optional[Callable] = None,
                 goal_shape: Optional[Iterable[int]] = None,
                 state: str = "obs",
                 action: str = "act",
                 next_state: str = "next_obs",
                 strategy: str = "future",
                 additional_goals: int = 4,
                 prioritized = True,
                 **kwargs):
        r"""
        Initialize ``HindsightReplayBuffer``

        Parameters
        ----------
        size : int
            Buffer Size
        env_dict : dict of dict
            Dictionary specifying environments. The keys of ``env_dict`` become
            environment names. The values of ``env_dict``, which are also ``dict``,
            defines ``"shape"`` (default ``1``) and ``"dtypes"`` (fallback to
            ``default_dtype``)
        max_episode_len : int
            Maximum episode length.
        reward_func : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
            Batch calculation of reward function:
            :math:`\mathcal{S}\times \mathcal{A}\times \mathcal{G} \to \mathcal{R}`.
        goal_func : Callable[[np.ndarray], np.ndarray], optional
            Batch extraction function for goal from state:
            :math:`\mathcal{S}\to\mathcal{G}`.
            If ``None`` (default), identity function is used (goal = state).
        goal_shape : Iterable[int], optional
            Shape of goal. If ``None`` (default), state shape is used.
        state : str, optional
            State name in ``env_dict``. The default is ``"obs"``.
        action : str, optional
            Action name in ``env_dict``. The default is ``"act"``.
        next_state : str, optional
            Next state name in ``env_dict``. The default is ``"next_obs"``.
        strategy : {"future", "episode", "random", "final"}, optional
            Goal sampling strategy.
            ``"future"`` selects one of the future states in the same episode.
            ``"episode"`` selects states in the same episode.
            ``"random"`` selects from the all states in replay buffer.
            ``"final"`` selects the final state in the episode.
            For ``"final"`` strategy, ``additional_goals`` is ignored.
            The default is ``"future"``.
        additional_goals : int, optional
            Number of additional goals. The default is ``4``.
        prioritized : bool, optional
            Whether use Prioritized Experience Replay. The default is ``True``.
        """
        self.max_episode_len = max_episode_len
        self.reward_func = reward_func
        self.goal_func = goal_func or (lambda s: s)

        self.state = state
        self.action = action
        self.next_state = next_state

        self.strategy = strategy
        known_strategy = ["future", "episode", "random", "final"]
        if self.strategy not in known_strategy:
            raise ValueError(f"Unknown Strategy: {strategy}. " +
                             f"Known Strategies: {known_strategy}")

        self.additional_goals = additional_goals
        if self.strategy == "final":
            self.additional_goals = 1

        self.prioritized = prioritized

        if goal_shape:
            goal_dict = {**env_dict[state], "shape": goal_shape}
            self.goal_shape = np.array(goal_shape, ndmin=1)
        else:
            goal_dict = env_dict[state]
            self.goal_shape = np.array(env_dict[state].get("shape", 1), ndmin=1)
        RB = PrioritizedReplayBuffer if self.prioritized else ReplayBuffer
        self.rb = RB(size, {**env_dict, "rew": {}, "goal": goal_dict}, **kwargs)

        self.episode_rb = ReplayBuffer(self.max_episode_len, env_dict)

        self.rng = np.random.default_rng()


    def add(self, **kwargs):
        r"""Add transition(s) into replay buffer.

        Multple sets of transitions can be added simultaneously.

        Parameters
        ----------
        **kwargs : array like or float or int
            Transitions to be stored.
        """
        if self.episode_rb.get_stored_size() >= self.max_episode_len:
            raise ValueError("Exceed Max Episode Length")
        self.episode_rb.add(**kwargs)


    def sample(self, batch_size: int, **kwargs):
        r"""Sample the stored transitions randomly with speciped size

        Parameters
        ----------
        batch_size : int
            sampled batch size

        Returns
        -------
        sample : dict of ndarray
            Batch size of sampled transitions, which might contains
            the same transition multiple times.
        """
        return self.rb.sample(batch_size, **kwargs)


    def on_episode_end(self, goal):
        """
        Terminate the current episode and set hindsight goal

        Paremeters
        ----------
        goal : array-like
            Original goal state of this episode.
        """
        episode_len = self.episode_rb.get_stored_size()
        if episode_len == 0:
            return None

        trajectory = self.episode_rb.get_all_transitions()
        add_shape = (trajectory[self.state].shape[0], *self.goal_shape)

        goal = np.broadcast_to(np.asarray(goal), add_shape)
        rew = self.reward_func(trajectory[self.next_state],
                               trajectory[self.action],
                               goal)

        self.rb.add(**trajectory, goal=goal, rew=rew)

        if self.strategy == "future":
            idx = np.zeros((self.additional_goals, episode_len), dtype=np.int64)
            for i in range(episode_len):
                idx[:,i] = self.rng.integers(low=i, high=episode_len,
                                             size=self.additional_goals)
            for i in range(self.additional_goals):
                goal = self.goal_func(trajectory[self.next_state][idx[i]])
                rew = self.reward_func(trajectory[self.next_state],
                                       trajectory[self.action],
                                       goal)
                self.rb.add(**trajectory, rew=rew, goal=goal)
        elif self.strategy == "episode":
            idx = self.rng.integers(low=0, high=episode_len,
                                    size=(self.additional_goals, episode_len))
            for _i in idx:
                goal = self.goal_func(trajectory[self.next_state][_i])
                rew = self.reward_func(trajectory[self.next_state],
                                       trajectory[self.action],
                                       goal)
                self.rb.add(**trajectory, rew=rew, goal=goal)
        elif self.strategy == "final":
            goal = self.goal_func(np.broadcast_to(trajectory[self.next_state][-1],
                                                  trajectory[self.next_state].shape))
            rew = self.reward_func(trajectory[self.next_state],
                                   trajectory[self.action],
                                   goal)
            self.rb.add(**trajectory, rew=rew, goal=goal)
        else: # random
            # Note 1:
            #   We should not prioritize goal selection,
            #   so that we manually create indices.
            # Note 2:
            #   Since we cannot access internal data directly,
            #   we have to extract set of transitions.
            #   Although this has overhead, it is fine
            #   becaue "random" strategy is used only for
            #   strategy comparison.
            idx = self.rng.integers(low=0,
                                    high=self.rb.get_stored_size(),
                                    size=self.additional_goals*episode_len)
            goal = self.goal_func(self.rb._encode_sample(idx)[self.next_state])
            goal = goal.reshape((self.additional_goals,
                                 episode_len,
                                 *(goal.shape[1:])))
            for g in goal:
                rew = self.reward_func(trajectory[self.next_state],
                                       trajectory[self.action],
                                       g)
                self.rb.add(**trajectory, rew=rew, goal=g)

        self.episode_rb.clear()
        self.rb.on_episode_end()

    def clear(self):
        """
        Clear replay buffer
        """
        self.rb.clear()
        self.episode_rb.clear()


    def get_stored_size(self):
        """
        Get stored size

        Returns
        -------
        int
            stored size
        """
        return self.rb.get_stored_size()


    def get_buffer_size(self):
        """
        Get buffer size

        Returns
        -------
        int
            buffer size
        """
        return self.rb.get_buffer_size()


    def get_all_transitions(self, shuffle: bool = False):
        r"""
        Get all transitions stored in replay buffer.

        Parameters
        ----------
        shuffle : bool, optional
            When ``True``, transitions are shuffled. The default value is ``False``.

        Returns
        -------
        transitions : dict of numpy.ndarray
            All transitions stored in this replay buffer.
        """
        return self.rb.get_all_transitions(shuffle)

    def update_priorities(self, indexes, priorities):
        """
        Update priorities

        Parameters
        ----------
        indexes : array_like
            indexes to update priorities
        priorities : array_like
            priorities to update

        Raises
        ------
        TypeError: When ``indexes`` or ``priorities`` are ``None``
        ValueError: When this buffer is constructed with ``prioritized=False``
        """
        if not self.prioritized:
            raise ValueError("Buffer is constructed without PER")

        self.rb.update_priorities(indexes, priorities)

    def get_max_priority(self):
        """
        Get max priority

        Returns
        -------
        float
            Max priority of stored priorities

        Raises
        ------
        ValueError: When this buffer is constructed with ``prioritized=False``
        """
        if not self.prioritized:
            raise ValueError("Buffer is constructed without PER")

        return self.rb.get_max_priority()
