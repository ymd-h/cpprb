from functools import reduce
import numpy as np
from scipy.special import softmax

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

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

class SoftmaxPolicy:
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
                 optimizer = Adam(),
                 **kwargs):
        self.env = env
        self.gamma = gamma
        self.prioritized = prioritized
        self.optimizer = optimizer

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

        self.model.compile(loss = "huber_loss",
                           optimizer = self.optimizer,
                           metrics=['accuracy'])
        self.target_model = clone_model(self.model)


    def sync_models(self):
        self.target_model.set_weights(self.model.get_weights())


    def train(self,batch_size = 256,*,
              callbacks = None):
        sample = self.buffer.sample(batch_size)
        obs = sample["obs"]

        predict_Q = self.model.predict(obs)
        target_Q = self.target_model.predict(sample["next_obs"]).max(axis=1) * (1.0 - sample["done"]) * self.gamma + sample["rew"]

        target_Q = tf.where(tf.one_hot(sample["act"],self.act_dim,True,False),
                            target_Q,predict_Q)

        if self.prioritized:
            TD = np.square(target_Q - predict_Q)
            self.buffer.update_priorities(sample["indexes"],TD)

        self.model.fit(x=obs,
                       y=target_Q,
                       batch_size=batch_size,
                       epochs = 1,
                       callbacks = callbacks)


    def __call__(self,policy,n_iteration,*,
                 batch_size = 256,
                 validation = 5,
                 callbacks = None,
                 local_buffer = 10,
                 longest_step = 500,
                 rew_func = None,
                 callback = None,
                 use_total_loss = True):

        explore(self.buffer,policy,n_iteration,
                local_buffer = local_buffer,
                longest_step = longest_step,
                rew_func = rew_func,
                callback = lambda it: self.train(batch_size,callbacks=callbacks))
