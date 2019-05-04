from functools import reduce
import numpy as np
from scipy.special import softmax

from tensorflow.keras import Sequential, clone_model
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizer import Adam

from cpprb import explore, create_buffer

class RandomPolicy:
    """Functor class which randomly select action from discrete action space.
    """
    def __init__(self,env,*args,**kwargs):
        """Initialize action dimension (act_dim) from environment (env)

        Parameters
        ----------
        env: gym.Env
            gym.Env compatible discrete action environment
        """
        self.act_dim = env.action_space.n

    def __call__(self,*args,**kwargs):
        """Return random action

        Returns
        -------
        : int
            selected action (random choice)
        """
        return np.random.choice(self.act_dim)

class GreedyPolicy:
    """Functor class which always select best prediction.
    """
    def __init__(self,model,*args,**kwargs):
        """Initialize prediction model.

        Parameters
        ----------
        model: tensorflow.keras.models.Model
            model to use prediction.
        """
        self.model = model

    def __call__(self,obs,*args,**kwargs):
        """Return best predicted action

        Paremeters
        ----------
        obs: array-like
            observation for prediction

        Returns
        -------
        : int
            selected action (best prediction)
        """
        return np.argmax(self.model.predict(obs.reshape(1,-1),batch_size=1))

class EpsilonGreedyPolicy(RandomPolicy,GreedyPolicy):
    """Functor class which almost select best prediction and sometime randomly.

    Within small epsilon fraction, select action randomly.
    """
    def __init__(self,env,model,eps = 0.1,*args,**kwargs):
        """Initialize action dimension (act_dim) and prediction model

        Parameters
        ----------
        env: gym.Env
            gym.Env compatible discrete action environment
        model: tensorflow.keral.models.Model
            model for prediction
        eps: float, optional
            small fraction for random selection
        """
        RandomPolicy.__init__(self,env,*args,**kwargs)
        GreedyPolicy.__init__(self,model,*args,**kwargs)
        self.eps = eps

    def __call__(self,obs,*args,**kwargs):
        """Return selected action

        Parameters
        ----------
        obs: array-like
            observation for prediction

        Returns
        -------
        : int
            selected action (best prediction, sometime random)
        """
        if np.random.rand() < eps:
            RandomPolicy.__call__(self,obs,*args,**kwargs)
        else:
            GreedyPolicy.__call__(self,obs,*args,**kwargs)

class SofmaxPolicy:
    """Functor class which select action with respect to softmax probabilities.
    """
    def __init__(self,model,*args,**kwargs):
        """Initialize prediction model

        Parameters
        ----------
        model: tensorflow.keras.models.Model
            model for predction
        """
        self.model = model

    def __call__(self,obs,*args,**kwargs):
        """Return selected action with regard to softmax probabilities.

        Parameters
        ----------
        obs: array-like
            observation for prediction

        Returns
        -------
        : int
            selected action with regard to softmax probabilities.
        """
        actions = softmax(np.ravel(model.predict(obs.reshape(1,-1),batch_size=1)))
        actions /= actions.sum()

        return np.random.choice(actions.shape[0],p=actions)

class DQN:
    def __init__(self,env, hidden_units = (64,64),*,
                 buffer_size = 1e6,
                 obs_shape = None,
                 prioritized = False,
                 Nstep = False,
                 process_shared = False,
                 *args,**kwargs):
        self.env = env

        self.obs_shape = self.env.observation_space.shape
        self.act_dim = self.env.action_space.n

        self.buffer = create_buffer(buffer_size,
                                    obs_shape = self.obs_shape,
                                    act_dim = self.act_dim,
                                    prioritized = prioritized,
                                    Nstep = Nstep,
                                    process_shared = process_shared)

        self.model = Sequential([InputLayer(input_shape=(self.obs_shape))])

        for units in hidden_units:
            self.model.add(Dense(units,activation="relu"))

        self.model.add(Dense(self.act_dim))

        self.model.compile()
        self.target_model = clone_model(self.model)


    def sync_models(self):
        self.target_model.set_weights(self.model.get_weights())


    def train(self,policy):
        pass
