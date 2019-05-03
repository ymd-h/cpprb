import numpy as np

from cpprb import explore

class RandomPolicy:
    def __init__(self,env,*args,**kwargs):
        self.act_dim = env.action_space.n

    def __call__(self,*args,**kwargs):
        return np.random.choices(self.act_dim)

class GreedyPolicy:
    def __init__(self,model,*args,**kwargs):
        self.model = model

    def __call__(self,obs,*args,**kwargs):
        return np.argmax(self.model.predict(obs))
