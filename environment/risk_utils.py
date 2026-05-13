import networkx as nx
import numpy as np

from .maps import graph_utils

import functools
from enum import IntEnum

from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces.utils import flatten, unflatten, flatdim
from pettingzoo.utils.wrappers import BaseWrapper

# State/observation spaces info:
#   territory_owner: The owner of the territory, -1 if owned by nobody.
#   number_of_armies: Number of armies in a territory.
#   action_phase: Number representing the phase (0: initial placing, 1: select army count, 2: select node, 3: select edge)
#   troops_to_place: How many reinforcement troops are left to place (if reinforcement phase).
#   selected_node: Node ID selected in a previous select_node move, -1 if not selected.
#   selected_edge: Edge ID selected in a previous select_edge move, -1 if not selected.

# Action space info:
#   The action is a single number representing, based on the current action_phase, either:
#   - Node ID to pick (if phase 0 or 2, usually meant for reinforcements)
#   - Edge ID to pick (if phase 3, usually meant for movement/attack)
#   - Number of armies to use (if phase 1, usually after a node/edge pick move)

class RiskPhase(IntEnum):
    STARTING_PLACEMENT = 0
    SELECT_ARMY_COUNT = 1
    SELECT_NODE = 2
    SELECT_EDGE = 3


@functools.lru_cache(maxsize=100)
def generic_observation_space(game_map : nx.DiGraph, num_agents : int, max_armies : int) -> Dict:
    
    return Dict(
        {
            "territory_owner": Box(low=-1, high=num_agents, shape=(game_map.number_of_nodes(), ), dtype=np.int8),
            "number_of_armies": Box(low=0, high=max_armies+1, shape=(game_map.number_of_nodes(), ), dtype=np.int16),
            "action_phase" : Discrete(len(RiskPhase), dtype=np.int8),
            "troops_to_place" : Discrete(max_armies, dtype=np.int16),
            "selected_node": Discrete(game_map.number_of_nodes()+1, start=-1, dtype=np.int16),
            "selected_edge": Discrete(game_map.number_of_edges()+1, start=-1, dtype=np.int16),
        }
    )


@functools.lru_cache(maxsize=100)
def generate_starting_observation(game_map : nx.DiGraph, num_agents : int, full_knowledge : bool = False) -> dict:

    return {
        "territory_owner" : np.full(shape=game_map.number_of_nodes(), fill_value=-1, dtype=np.int8),
        "number_of_armies" : np.zeros(game_map.number_of_nodes(), dtype=np.int16),
        "action_phase" : np.array(0, dtype=np.int8),
        "troops_to_place" : np.array(0, dtype=np.int16),
        "selected_node": np.array(-1, dtype=np.int16),
        "selected_edge": np.array(-1, dtype=np.int16)
    }


@functools.lru_cache(maxsize=100)
def generic_action_space(game_map : nx.DiGraph, num_agents : int, max_armies : int) -> Dict:

    return Discrete(max(game_map.number_of_nodes(), game_map.number_of_edges(), max_armies), dtype=np.int16)


@functools.lru_cache(maxsize=100)
def generic_mask_space(game_map : nx.DiGraph, num_agents : int, max_armies : int) -> Dict:

    return Box(low=0, high=1, shape=(max(game_map.number_of_nodes(), game_map.number_of_edges(), max_armies), ), dtype=np.int8)


def generate_action_mask(game_map : nx.DiGraph, max_armies : int, agent_state : dict, agent_id : int) -> np.ndarray:

    mask = np.zeros([max(game_map.number_of_nodes(), game_map.number_of_edges(), max_armies)], dtype=np.int8)
    owned_nodes = set(np.where(agent_state["territory_owner"] == agent_id)[0].tolist())

    match agent_state["action_phase"]:

        case RiskPhase.STARTING_PLACEMENT:
            # Still placing at the start, can place one in ANY UNOWNED territories.
            mask[np.where(agent_state["territory_owner"] == -1)] = True

        case RiskPhase.SELECT_NODE:
            # Since the only time one can pick a node that is NOT the starting placement phase is to reinforce, one can pick any owned nodes.
            mask[:agent_state["territory_owner"].size] = agent_state["territory_owner"] == agent_id

        case RiskPhase.SELECT_EDGE:

            for node in owned_nodes:
                if agent_state["number_of_armies"][node] <= 1:
                    # Node only has one army, can't move or attack.
                    continue
                for edge in game_map.out_edges(game_map.graph["idx_to_node"][node]):
                    mask[game_map.graph["edge_to_idx"][edge]] = 1

        case RiskPhase.SELECT_ARMY_COUNT:

            if agent_state["troops_to_place"] > 0:
                # Still has reinforcement troops to place. Assuming a node has already been picked.
                if agent_state["selected_node"] == -1:
                    raise RuntimeError(f"Inconsistent world space: selected_node is -1.")
                mask[1:agent_state["troops_to_place"]+1] = 1
            else:
                # No reinforcement troops means this is likely an attack/move action.
                if agent_state["selected_edge"] == -1:
                    raise RuntimeError(f"Inconsistent world space: selected_edge is -1.")
                edge = game_map.graph["idx_to_edge"][agent_state["selected_edge"]]
                source_node_armies = agent_state["number_of_armies"][game_map.graph["node_to_idx"][edge[0]]]
                if game_map.graph["node_to_idx"][game_map.graph["idx_to_edge"][agent_state["selected_edge"]][1]] in owned_nodes:
                    # This is a movement action.
                    mask[1:source_node_armies] = 1
                else:
                    # This is an attack action.
                    mask[1:min(source_node_armies, 4)] = 1

    return mask


def risk_attack_outcome(atk_armies : int, def_armies : int) -> tuple[int, int]:
    # Given a number of attackers and defenders, returns the number of losses on both sides.
    atk_dice_amount = min(3, atk_armies)
    def_dice_amount = min(2, def_armies)

    atk_roll = np.random.randint(1, 7, size=atk_dice_amount)
    def_roll = np.random.randint(1, 7, size=def_dice_amount)

    atk_roll.sort()
    def_roll.sort()

    losses_att = 0
    losses_def = 0
    for a, d in zip(reversed(atk_roll), reversed(def_roll)):
        if a > d:
            # Attacker has a higher die roll.
            losses_def += 1
        else:
            # Defender has a higher or equal die roll.
            losses_att += 1

    return losses_att, losses_def