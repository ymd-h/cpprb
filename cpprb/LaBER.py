import numpy as np


class LaBER:
    def __init__(self, batch_size: int, m: int = 4, *, eps: float = 1e-6):
        """
        Initialize LaBER (sub-)class

        Parameters
        ----------
        batch_size : int
            Batch size for neural network
        m : int, optional
            Multiplication factor. ``m * batch_size`` transitions will be passed.
            Default value is ``4``.
        eps : float, option
            Small positive values to avoid 0 priority. Default value is ``1e-6``.


        Raises
        ------
        ValueError
            When ``batch_size <= 0``, ``m <= 0``, or ``eps <= 0``.
        """
        self.rng = np.random.default_rng()

        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("``batch_size`` must be positive integer.")

        if m <= 0:
            raise ValueError("``m`` must be positive integer")

        self.idx = np.arange(int(self.batch_size * m))

        self.eps = float(eps)
        if self.eps < 0:
            raise ValueError("``eps`` must be non negative")

    def __call__(self, *, priorities, **kwargs):
        """
        Sub-sample from large batch

        Parameters
        ----------
        priorities : array-like of float
            Surrogate priorities.
        **kwargs : key-value
            Large batch sampled from ``ReplayBuffer``. These values are also
            included sub-sampled batch.

        Returns
        -------
        dict
            Sub-sampled batch, which includes ``"weights"``, ``"indexes"``,
            and passed keys.

        Raises
        ------
        ValueError
            If the size of ``priorities`` is not ``batch_size * m``.
        """
        p = np.asarray(priorities) + self.eps
        if p.shape != self.idx.shape:
            raise ValueError("``priorities`` size must be ``batch_size * m``")

        p = p / p.sum()

        _idx = self.rng.choice(self.idx, self.batch_size, p=p)

        kwargs = {} if (kwargs is None) else {k: v[_idx] for k, v in kwargs.items()}

        kwargs["weights"] = self._normalize_weight(p, _idx)
        kwargs["indexes"] = _idx

        return kwargs

    def _normalize_weight(self, p, _idx):
        raise NotImplementedError


class LaBERmean(LaBER):
    """
    Helper class for Large Batch Experience Replay (LaBER)

    This helper class is a functor designed to be used together with ``ReplayBuffer``.
    It takes surrogate priorities for large batch, then returns sub-sampled indexes
    and weights.

    See Also
    --------
    LaBERmax, LaBERlazy : Other variants

    Notes
    -----
    In LaBER [1]_, first m-times larger batch (large bacth) is sampled from
    Replay Buffer. The final mini-batch is sampled from the large batch based on
    newly calculated surrogate priorities.
    This class implements LaBER-mean variant, where weights are normalized by
    average over the large batch.

    References
    ----------
    .. [1] T. Lahire et al, "Large Batch Experience Replay", CoRR (2021)
       https://dblp.org/db/journals/corr/corr2110.html#journals/corr/abs-2110-01528
       https://arxiv.org/abs/2110.01528
    """

    def _normalize_weight(self, p, _idx):
        return p.mean() / p[_idx]


class LaBERlazy(LaBER):
    """
    Helper class for Large Batch Experience Replay (LaBER)

    This helper class is a functor designed to be used together with ``ReplayBuffer``.
    It takes surrogate priorities for large batch, then returns sub-sampled indexes
    and weights.

    Warnings
    --------
    According to the proposed paper [1]_, ``LaBERmean`` is more preferable.

    See Also
    --------
    LaBERmean, LaBERmax : Other variants

    Notes
    -----
    In LaBER [1]_, first m-times larger batch (large bacth) is sampled from
    Replay Buffer. The final mini-batch is sampled from the large batch based on
    newly calculated surrogate priorities.
    This class implements LaBER-lazy variant, where weights are not normalized at all.

    References
    ----------
    .. [1] T. Lahire et al, "Large Batch Experience Replay", CoRR (2021)
       https://dblp.org/db/journals/corr/corr2110.html#journals/corr/abs-2110-01528
       https://arxiv.org/abs/2110.01528
    """

    def _normalize_weight(self, p, _idx):
        return 1.0 / p[_idx]


class LaBERmax(LaBER):
    """
    Helper class for Large Batch Experience Replay (LaBER)

    This helper class is a functor designed to be used together with ``ReplayBuffer``.
    It takes surrogate priorities for large batch, then returns sub-sampled indexes
    and weights.

    Warnings
    --------
    According to the proposed paper [1]_, ``LaBERmean`` is more preferable.

    See Also
    --------
    LaBERmean, LaBERlazy : Other variants

    Notes
    -----
    In LaBER [1]_, first m-times larger batch (large bacth) is sampled from
    Replay Buffer. The final mini-batch is sampled from the large batch based on
    newly calculated surrogate priorities.
    This class implements LaBER-max variant, where weights are normalized by
    the maximum weight of selected mini-batch.

    References
    ----------
    .. [1] T. Lahire et al, "Large Batch Experience Replay", CoRR (2021)
       https://dblp.org/db/journals/corr/corr2110.html#journals/corr/abs-2110-01528
       https://arxiv.org/abs/2110.01528
    """

    def _normalize_weight(self, p, _idx):
        p_idx = 1.0 / p[_idx]
        return p_idx / p_idx.max()
