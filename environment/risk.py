# https://gymnasium.farama.org/api/env/
# https://gymnasium.farama.org/api/spaces/
# https://pettingzoo.farama.org/content/environment_creation/

from copy import copy
import functools
import random

import numpy as np
import networkx as nx

import gymnasium
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
        self.territory_armies = {}
        for territory in list(self.map_network.nodes):
            self.territory_armies[territory] = [None for _ in self.possible_agents]
        self.player_armies = [{} for _ in self.possible_agents]


    def reset(self, seed=None, options=None):

        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.territory_armies = {}
        for territory in list(self.map_network.nodes):
            self.territory_armies[territory] = [0 for _ in self.possible_agents]

        observation = {
            "ownership"     : np.full(G.number_of_nodes(), -1, dtype=np.int32),
            "troops"        : np.zeros(G.number_of_nodes(), dtype=np.int32),
            "adjacency"     : nx.to_numpy_array(self.map_network).astype(np.int32),
            "phase"         : np.array([0], dtype=np.int32)
        }

        observations = {}
        for agent in self.agents:
            observations[agent] = {
                "observation" : observation,
                "action_mask" : ...
            }



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


    def step(self, action):
        pass