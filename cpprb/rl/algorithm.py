import numpy as np
from scipy.special import softmax

from cpprb import explore

class RandomPolicy:
    def __init__(self,env,*args,**kwargs):
        self.act_dim = env.action_space.n

    def __call__(self,*args,**kwargs):
        return np.random.choice(self.act_dim)

class GreedyPolicy:
    def __init__(self,model,*args,**kwargs):
        self.model = model

    def __call__(self,obs,*args,**kwargs):
        return np.argmax(self.model.predict(obs))

class EpsilonGreedyPolicy(RandomPolicy,GreedyPolicy):
    def __init__(self,env,model,eps = 0.1,*args,**kwargs):
        RandomPolicy.__init__(self,env,*args,**kwargs)
        GreedyPolicy.__init__(self,model,*args,**kwargs)
        self.eps = eps

    def __call__(self,obs,*args,**kwargs):
        if np.random.rand() < eps:
            RandomPolicy.__call__(self,obs,*args,**kwargs)
        else:
            GreedyPolicy.__call__(self,obs,*args,**kwargs)

class SofmaxPolicy:
    def __init__(self,model,*args,**kwargs):
        self.model = model

    def __call__(self,obs,*args,**kwargs):
        actions = softmax(np.ravel(model.predict(obs.reshape(1,-1),batch_size=1)))
        actions /= actions.sum()

        return np.random.choice(actions.shape[0],p=actions)
