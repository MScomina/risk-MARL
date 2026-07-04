from gymnasium.spaces import Box, Dict, Discrete, Space

from pettingzoo import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper

import numpy as np

class FlattenObservationWrapper(BaseWrapper):
    
    def __init__(self, env: AECEnv):
        super().__init__(env)
        
        self._observation_spaces = {}
        
        for agent in env.possible_agents:
            orig_space = env.observation_space(agent)
            
            if isinstance(orig_space, Dict) and "observation" in orig_space.spaces:
                sub_space = orig_space.spaces["observation"]
                flat_dim = 0
                
                for key, subspace in sub_space.spaces.items():
                    if isinstance(subspace, Box):
                        flat_dim += int(np.prod(subspace.shape))
                    elif isinstance(subspace, Discrete):
                        flat_dim += 1
                    else:
                        raise TypeError(f"Unsupported sub-space type: {type(subspace)}")
                
                flat_obs_space = Box(low=-1, high=32767, shape=(flat_dim,), dtype=np.int16)
                
                self._observation_spaces[agent] = Dict({
                    "observation": flat_obs_space,
                    "action_mask": orig_space.spaces["action_mask"]
                })
            else:
                self._observation_spaces[agent] = orig_space

    def observation_space(self, agent) -> Space:
        return self._observation_spaces[agent]

    def observe(self, agent) -> dict:
        orig_obs = self.env.observe(agent)
        if orig_obs is None:
            return None
            
        obs_dict = orig_obs["observation"]
        flat_parts = []
        
        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray):
                flat_parts.append(value.ravel())
            else:
                flat_parts.append(np.array([value], dtype=np.int16))
                
        output_dict = {
            "observation": np.concatenate(flat_parts).astype(np.int16),
            "action_mask": orig_obs["action_mask"]
        }
        return output_dict