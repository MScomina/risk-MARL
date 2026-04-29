# https://gymnasium.farama.org/api/env/
# https://gymnasium.farama.org/api/spaces/
# https://pettingzoo.farama.org/content/environment_creation/
# Inspiration for environment (masking): https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/chess/chess.py

from copy import copy
import functools
import random

import numpy as np
import networkx as nx

import gymnasium
from gymnasium.utils import seeding
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, MultiBinary
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from pathlib import Path

from maps import graph_utils

# Environment definitions:
#   (Underlying) Space states:  
#       - Ownership of territories (part of observation)
#       - Amount of armies in each territory (part of observation)
#       - Whether it's the starting placing step or not (part of observation)
#       - Phase (whether it's fortifying or attack/move phase) (part of observation)
#       - Cards in hand (partial observability of each agent) (TBI)
#   Actions:
#       If still starting placing: place until every territory is filled by somebody.
#       Otherwise:
#       - If phase 0: place any amount of troops on nodes of graph until no more can be placed
#       - If phase 1: pick any two (connected) nodes (owned by the player) and number of armies:
#           - If only one owned: Attack (max 3)
#           - If both are owned: Move, then end turn

MAX_ITERS = 10_000

class raw_env(AECEnv):

    metadata : dict = {"render_modes": ["human"], "name": "risk-v1"}

    def __init__(self, render_mode=None, n_agents : int = 2, map_path : Path | None = None, max_armies : int = 100):

        self.possible_agents = ["player_" + str(r) for r in range(n_agents)]
        self.map_network = graph_utils.generate_graph(map_path).to_directed()
        self.max_armies = max_armies
        self._observation_spaces = Dict(
            {
                agent: Dict(
                    {
                        "owned": MultiBinary(self.map_network.number_of_nodes()),
                        "territory_owner": MultiDiscrete([n_agents+1] * self.map_network.number_of_nodes()),
                        "number_of_armies": MultiDiscrete([max_armies+1] * self.map_network.number_of_nodes()),
                        "is_starting_placement" : Discrete(2),
                        "troops_to_place" : Discrete(self.max_armies)
                    } for agent in self.possible_agents
                )
            }
        )
        self._action_spaces = Dict(
            {
                agent: Dict(
                    {
                        "reinforce_move" : MultiDiscrete(self.map_network.number_of_nodes(), max_armies+1),
                        "atk_move" : Dict(
                            {
                                "src" : Discrete(self.map_network.number_of_nodes()),
                                "dst" : Discrete(self.map_network.number_of_nodes()),
                                "amount" : Discrete(self.max_armies),
                                "is_move" : Discrete(2)
                            }
                        )
                    }
                ) for agent in self.possible_agents
            }
        )

        self.render_mode = render_mode


    @functools.lru_cache(maxsize=1000000)
    def observation_space(self, agent):
        return self._observation_spaces[agent]


    @functools.lru_cache(maxsize=1000000)
    def action_space(self, agent):
        return self._action_spaces[agent]


    def reset(self, seed=None, options=None):

        self.timestep = 0

        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        self.agents = copy(self.possible_agents)
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.state = {
            agent : {
                "owned" : np.full(shape=self.map_network.number_of_nodes(), fill_value=False),
                "territory_owner" : np.full(shape=self.map_network.number_of_nodes(), fill_value=self.num_agents),
                "number_of_armies" : np.zeros(self.map_network.number_of_nodes()),
                "is_starting_placement" : 1,
                "troops_to_place" : 0
            } for agent in self.agents
        }
        self.observations = {
            agent : {
                "owned" : np.full(shape=self.map_network.number_of_nodes(), fill_value=False),
                "territory_owner" : np.full(shape=self.map_network.number_of_nodes(), fill_value=self.num_agents),
                "number_of_armies" : np.zeros(self.map_network.number_of_nodes()),
                "is_starting_placement" : 1,
                "troops_to_place" : 0
            } for agent in self.agents
        }

        self.num_moves = 0

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()


    def observe(self, agent):
        return self.observations[agent]


    def close(self):
        pass


    def step(self, action):
        
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        self._cumulative_rewards[agent] = 0
        if self._agent_selector.is_last():
            # TODO: Rewarding.

            self.num_moves += 1
            self.truncations = {
                agent: self.num_moves >= MAX_ITERS for agent in self.agents
            }
