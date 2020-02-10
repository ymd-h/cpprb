import numpy as np

from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict
from gym.spaces import flatdim, flatten, unflatten

def from_space(space,int_type,float_type):
    if isinstance(space,Discrete):
        return {"dtype": int_type,"shape": 1}
    elif isinstance(space,MultiDiscrete):
        return {"dtype": int_type,"shape": space.nvec.shape}
    elif isinstance(space,Box):
        return {"dtype": float_type,"shape": space.shape}
    elif isinstance(space,MultiBinary):
        return {"dtype": int_type, "shape": space.n}
    else:
        raise NotImplementedError(f"Error: Unknown Space {space}")

def create_env_dict(env,*,int_type = None,float_type = None):
    """
    Create `env_dict` from Open AI `gym.space` for `ReplayBuffer.__init__`

    Paremeters
    ----------
    env : gym.Env
        Environment
    int_type: np.dtype, optional
        Integer type. Default is `np.int32`
    float_type: np.dtype, optional
        Floating point type. Default is `np.float32`

    Returns
    -------
    env_dict : dict
        env_dict parameter for `ReplayBuffer` class.
    """

    int_type = int_type or np.int32
    float_type = float_type or np.float32

    env_dict = {"rew" : {"shape": 1, "dtype": float_type},
                "done": {"shape": 1, "dtype": float_type}}

    observation_space = env.observation_space
    action_space = env.action_space

    if isinstance(observation_space,Tuple):
        for i,s in enumerate(observation_space.spaces):
            env_dict[f"obs{i}"] = from_space(s,int_type,float_type)
            env_dict[f"next_obs{i}"] = from_space(s,int_type,float_type)
    elif isinstance(observation_space,Dict):
        for n, s in observation_space.spaces.items():
            env_dict[n] = from_space(s,int_type,float_type)
            env_dict[f"next_{n}"] = from_space(s,int_type,float_type)
    else:
        env_dict["obs"] = from_space(observation_space,int_type,float_type)
        env_dict["next_obs"] = from_space(observation_space,int_type,float_type)

    if isinstance(action_space,Tuple):
        for i,s in enumerate(action_space.spaces):
            env_dict[f"act{i}"] = from_space(s,int_type,float_type)
    elif isinstance(action_space,Dict):
        for n, s in action_space.spaces.items():
            env_dict[n] = from_space(s,int_type,float_type)
    else:
        env_dict["act"] = from_space(action_space,int_type,float_type)

    return env_dict

def create_before_add_func(env):
    """
    Create function to be used before `ReplayBuffer.add`

    Parameters
    ----------
    env : gym.Env
        Environment for before_func

    Returns
    -------
    before_add : callable
        Function to be used before `ReplayBuffer.add`
    """
    def no_convert(name,v):
        return {f"{name}": v}

    def convert_from_tuple(name,_tuple):
        return {f"{name}{i}": v for i,v in enumerate(_tuple)}

    def convert_from_dict(name,_dict):
        return {f"{name}_{key}":v for key,v in _dict.items()}


    observation_space = env.observation_space
    action_space = env.action_space


    if isinstance(observation_space,Tuple):
        obs_func = convert_from_tuple
    elif isinstance(observation_space,Dict):
        obs_func = convert_from_dict
    else:
        obs_func = no_convert

    if isinstance(action_space,Tuple):
        act_func = convert_from_tuple
    elif isinstance(action_space,Dict):
        act_func = convert_from_dict
    else:
        act_func = no_convert

    def before_add(obs,act,next_obs,rew,done):
        return {**obs_func("obs",obs),
                **act_func("act",act),
                **obs_func("next_obs",next_obs),
                "rew": rew,
                "done": done}

    return before_add
