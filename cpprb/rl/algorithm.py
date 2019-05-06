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
    def __init__(self,*args,**kwargs):
        """Initialize action dimension (act_dim) from environment (env)

        Parameters
        ----------
        env: gym.Env
            gym.Env compatible discrete action environment
        """
        pass

    def __call__(self,model,obs,obs_shape,act_dim,*args,**kwargs):
        """Return random action

        Returns
        -------
        : int
            selected action (random choice)
        """
        return np.random.choice(act_dim)

class GreedyPolicy:
    """Functor class which always select best prediction.
    """
    def __init__(self,*args,**kwargs):
        """Initialize prediction model.

        Parameters
        ----------
        model: tensorflow.keras.models.Model
            model to use prediction.
        """
        pass

    def __call__(self,model,obs,obs_shape,act_dim,*args,**kwargs):
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
        return np.argmax(model.predict(obs.reshape(1,*obs_shape)))

class EpsilonGreedyPolicy(RandomPolicy,GreedyPolicy):
    """Functor class which almost select best prediction and sometime randomly.

    Within small epsilon fraction, select action randomly.
    """
    def __init__(self,eps = 0.1,*args,**kwargs):
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
        RandomPolicy.__init__(self,*args,**kwargs)
        GreedyPolicy.__init__(self,*args,**kwargs)
        self.eps = eps

    def __call__(self,*args,**kwargs):
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
            RandomPolicy.__call__(self,*args,**kwargs)
        else:
            GreedyPolicy.__call__(self,*args,**kwargs)

class SofmaxPolicy:
    """Functor class which select action with respect to softmax probabilities.
    """
    def __init__(self,*args,**kwargs):
        """Initialize prediction model

        Parameters
        ----------
        model: tensorflow.keras.models.Model
            model for predction
        """
        pass

    def __call__(self,model,obs,obs_shape,act_dim,*args,**kwargs):
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
        actions = softmax(np.ravel(model.predict(obs.reshape(1,*obs_shape))))
        actions /= actions.sum()

        return np.random.choice(actions.shape[0],p=actions)

class DQN:
    def __init__(self,env, hidden_units = (64,64),*,
                 buffer_size = 1e6,
                 prioritized = False,
                 Nstep = False,
                 process_shared = False,
                 gamma = 0.99,
                 *args,**kwargs):
        self.env = env
        self.gamma = gamma
        self.prioritized = prioritized

        self.obs_shape = self.env.observation_space.shape
        self.act_dim = self.env.action_space.n

        self.buffer = create_buffer(buffer_size,
                                    obs_shape = self.obs_shape,
                                    act_dim = self.act_dim,
                                    prioritized = self.prioritized,
                                    Nstep = Nstep,
                                    process_shared = process_shared,
                                    is_discrete_action = True)

        self.model = Sequential([InputLayer(input_shape=(self.obs_shape))])

        for units in hidden_units:
            self.model.add(Dense(units,activation="relu"))

        self.model.add(Dense(self.act_dim))

        self.model.compile()
        self.target_model = clone_model(self.model)


    def sync_models(self):
        self.target_model.set_weights(self.model.get_weights())


    def train(self,batch_size = 256):
        sample = buffer.sample(batch_size)
        obs = sample["obs"]

        target_y = self.target_model.predict(sample["next_obs"]).max(axis=1) * (1.0 - sample["done"]) * self.gamma + sample["rew"]

        if self.prioritized:
            predict_y = self.model.predict(obs)[np.arange(batch_size),act]
            TD = np.square(target_y - predict_y)
            buffer.update_priorities(sample["indexes"],TD)

        self.model.fit(x=obs,
                       y=target_y,
                       batch_size=batch_size,
                       epoch = epoch,
                       callbacks = [])


    def __call__(self,policy,n_iteration,*
                 batch_size = 256,
                 validation = 5,
                 callbacks = None,
                 local_buffer = 10,
                 longest_step = 500,
                 rew_func = None,
                 callback = None):

        greedy = GreedyPolicy()

        explore(self.buffer,policy,n_iteration,
                local_buffer = local_buffer,
                longest_step = longest_step,
                rew_func = rew_func,
                callback = self.train)

        explore(self.buffer,greedy,validation,
                local_buffer = local_buffer,
                longest_step = longest_step,
                rew_func = rew_func,
                callback = callback)
