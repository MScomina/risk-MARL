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
                agent: risk_utils.generic_action_space(
                    self.map_network,
                    self.num_agents,
                    self.max_armies
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

        self.world_state = risk_utils.generate_starting_observation(
            game_map=self.map_network, 
            num_agents=self.num_agents
        )

        self.state = {
            agent : self.world_state for agent in self.agents
        }

        self.observations = {
            agent : {
                "observation": self.world_state,
                "action_mask": risk_utils.generate_action_mask(
                    game_map=self.map_network,
                    max_armies=self.max_armies,
                    agent_state=self.world_state,
                    agent_id=idx
                )
            } for idx, agent in enumerate(self.agents)
        }

        self.num_moves = 0

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()


    def observe(self, agent):
        return self.observations[agent]


    def close(self):
        pass


    def step(self, action):
        
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        current_agent = self.agent_selection
        current_index = self.agents.index(current_agent)

        self._cumulative_rewards[agent] = 0

        self._update_state(agent, action)
        if self._agent_selector.is_last():
            # TODO: Rewarding.



            self.num_moves += 1
            self.truncations = {
                agent: self.num_moves >= self.max_iters for agent in self.agents
            }


    def _update_state(self, agent : str, action : dict) -> bool:
        # Updates the state and returns whether the turn is over or not.
        is_reinforce = action["reinforce_move"][1] != 0
        is_attack = action["atk_move"][2]
        agent_idx = self.agents.index(agent)

        if is_reinforce:
            node = action["reinforce_move"][0]
            atk_amount = action["reinforce_move"][1]

            node_unowned = self.world_state["owned"][node] == 0
            owned_by_agent = self.world_state["territory_owner"][node] == agent_idx

            if not ((node_unowned and atk_amount == 1) or (owned_by_agent and atk_amount <= self.world_state["troops_to_place"])):
                raise ValueError(f"Illegal reinforce action performed by agent {agent}: {action}.")

            if node_unowned:
                # Claiming territory.
                self.world_state["territory_owner"][node] = agent_idx
                self.world_state["number_of_armies"][node] = 1
                self.world_state["owned"][node] = 1
            else:
                # Reinforcing already owned territory.
                self.world_state["number_of_armies"][node] += atk_amount
                self.world_state["troops_to_place"] -= atk_amount

        else:
            edge = action["atk_move"][0]
            atk_amount = action["atk_move"][1]
            src, dst = self.map_network.edge_to_nodes_id(edge)

            is_owner_src = self.world_state["territory_owner"][src] == agent_idx
            is_owner_dst = self.world_state["territory_owner"][dst] == agent_idx

            src_amount = self.world_state["number_of_armies"][src]

            if not (is_owner_src and ((is_attack and not is_owner_dst and atk_amount <= 3) or (not is_attack and is_owner_dst and atk_amount < src_amount))):
                raise ValueError(f"Illegal attack action performed by agent {agent}: {action}.")

            if is_attack:
                # Attacking opponent's territory.
                dst_amount = self.world_state["number_of_armies"][dst]
                atk_losses, def_losses = risk_utils.risk_attack_outcome(atk_amount, dst_amount)

                self.world_state["number_of_armies"][src] -= atk_losses
                self.world_state["number_of_armies"][dst] -= def_losses

                if self.world_state["number_of_armies"][dst] == 0:
                    # No troops left, control switches.
                    self.world_state["territory_owner"][dst] = agent_idx
                    self.world_state["number_of_armies"][dst] += atk_amount - atk_losses
                    self.world_state["number_of_armies"][src] -= atk_amount - atk_losses
            else:
                # Reinforcing one's territory from another territory.
                self.world_state["number_of_armies"][src] -= atk_amount
                self.world_state["number_of_armies"][dst] += atk_amount
                return True

        return False