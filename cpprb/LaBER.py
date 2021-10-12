import numpy as np

class LaBER:
    def __init__(self, batch_size: int, m: int = 4, *, eps: float = 1e-6):
        """
        Initialize LaBER (sub-)class

        Ref: https://arxiv.org/abs/2110.01528

        Parameters
        ----------
        batch_size : int
            Batch size for neural network
        m : int, optional
            Multiplication factor. `m * batch_size` transitions will be passed.
            Default value is `4`.
        eps : float, option
            Small positive values to avoid 0 priority. Default value is `1e-6`.
        """
        self.rng = np.random.default_rng()

        self.batch_size = int(batch_size)
        self.idx = np.arange(int(self.batch_size * m))

        self.eps = float(eps)

    def __call__(self, *, priorities, **kwargs):
        """
        Sub-sample from large batch

        Parameters
        ----------
        priorities : array-like of float
            Surrogate priorities.
        **kwargs : key-value
            Large batch sampled from `ReplayBuffer`.
        """
        p = np.asarray(priorities) + self.eps
        p = p / p.sum()

        _idx = self.rng.choice(self.idx, self.batch_size, p=p)

        kwargs = {k: v[_idx] for k, v in kwargs.items()}
        kwargs["weights"] = self._normalize_weight(p, _idx)
        kwargs["indexes"] = _idx

        return kwargs

    def _normalize_weight(self, p, _idx):
        raise NotImplementedError


def LaBERmean(LaBER):
    def _normalize_weight(self, p, _idx):
        return p.mean() / p[_idx]


def LaBERlazy(LaBER):
    def _normalize_weight(self, p, _idx):
        return 1.0 / p[_idx]


def LaBERmax(LaBER):
    def _normalize_weight(self, p, _idx):
        p_idx = 1.0 / p[_idx]
        return p_idx / p_idx.max()
