import numpy as np

from cpprb import explore

class RandomPolicy:
    def __init__(self,env):
        self.act_dim = env.action_space.n

    def __call__(self,*args,**kwargs):
        return np.random.choices(self.act_dim)
