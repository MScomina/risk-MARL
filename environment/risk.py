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
import risk_utils

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
#       - If phase 1: pick any edge and number of armies and action to perform.


# Default values
NUM_AGENTS = 2
MAX_ARMIES = 100
MAX_ITERS = 10_000


class raw_env(AECEnv):

    metadata : dict = {"render_modes": ["human"], "name": "risk-v1"}

    def __init__(self, render_mode=None, num_agents : int = NUM_AGENTS, map_path : Path | None = None, max_armies : int = MAX_ARMIES, max_iters : int = MAX_ITERS):

        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]
        self.num_agents = num_agents
        self.map_network = graph_utils.generate_graph(map_path)
        self.max_armies = max_armies
        self.max_iters = max_iters
        self._observation_spaces = Dict(
            {
                agent: risk_utils.generic_observation_space(
                    self.map_network,
                    self.num_agents,
                    self.max_armies
                ) for agent in self.possible_agents
            }
        )
        self._action_spaces = Dict(
            {
                agent: Dict(
                    {
                        "reinforce_move" : MultiDiscrete(self.map_network.number_of_nodes(), max_armies+1),     # Reinforce move (place troops).
                        "atk_move" : Dict(
                            {
                                "edge" : Discrete(self.map_network.number_of_edges()),  # Directed edge of graph where the movement/attack happens.
                                "amount" : Discrete(self.max_armies),                   # How many armies are used.
                                "is_move" : Discrete(2)                                 # Whether it's a movement or an attack.
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
            agent : risk_utils.generate_starting_observation(
                game_map=self.map_network, 
                num_agents=self.num_agents, 
                full_knowledge=True
                ) for agent in self.agents
        }

        self.observations = {
            agent : risk_utils.generate_starting_observation(
                game_map=self.map_network, 
                num_agents=self.num_agents
                ) for agent in self.agents
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
                agent: self.num_moves >= self.max_iters for agent in self.agents
            }
