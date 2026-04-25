# https://gymnasium.farama.org/api/env/
# https://gymnasium.farama.org/api/spaces/
# https://pettingzoo.farama.org/content/environment_creation/

import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from pathlib import Path

from maps import graph_utils

class raw_env(AECEnv):


    metadata : dict = {"render_modes": [], "name": "risk-v1"}


    def __init__(self, render_mode=None, n_agents : int = 2, map_path : Path | None = None):

        self.possible_agents = ["player_" + str(r) for r in range(n_agents)]
        self.map_network = graph_utils.generate_graph(map_path)



    @functools.lru_cache(maxsize=1000000)
    def observation_space(self, agent):
        pass


    @functools.lru_cache(maxsize=1000000)
    def action_space(self, agent):
        pass


    def observe(self, agent):
        pass


    def close(self):
        pass


    def reset(self, seed=None, options=None):
        pass


    def step(self, action):
        pass