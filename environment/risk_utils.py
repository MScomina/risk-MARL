import networkx as nx
import numpy as np

from gymnasium.spaces import Dict, Discrete, MultiDiscrete, MultiBinary

# State/observation spaces info:
#   owned: Whether a territory is owned or not by someone.
#   territory_owner: The owner of the territory, -1 if owned by nobody.
#   number_of_armies: Number of armies in a territory.
#   is_starting_placement: Whether it's the starting phase of placing troops on the board or not.
#   troops_to_place: How many reinforcement troops are left to place (if reinforcement phase).

# Action space info:
#   reinforce_move: Reinforce move (place troops on node).
#   atk_move:
#       edge: Directed edge of graph where the movement/attack happens.
#       amount: How many armies are used.
#       is_movement: Whether it's a movement or an attack.

def generic_observation_space(game_map : nx.DiGraph, num_agents : int, max_armies : int) -> Dict:
    
    return Dict(
        {
            "owned": MultiBinary(game_map.number_of_nodes()),
            "territory_owner": MultiDiscrete([num_agents+1] * game_map.number_of_nodes(), start=[-1] * game_map.number_of_nodes(), dtype=np.int8),
            "number_of_armies": MultiDiscrete([max_armies+1] * game_map.number_of_nodes(), dtype=np.int16),
            "is_starting_placement" : Discrete(2, dtype=np.int8),
            "troops_to_place" : Discrete(max_armies, dtype=np.int16)
        }
    )


def generate_starting_observation(game_map : nx.DiGraph, num_agents : int, full_knowledge : bool = False) -> dict:

    return {
        "owned" : np.full(shape=game_map.number_of_nodes(), fill_value=False, dtype=np.bool),
        "territory_owner" : np.full(shape=game_map.number_of_nodes(), fill_value=-1, dtype=np.int8),
        "number_of_armies" : np.zeros(game_map.number_of_nodes(), dtype=np.int16),
        "is_starting_placement" : 1,
        "troops_to_place" : 0
    }


def generic_action_space(game_map : nx.DiGraph, num_agents : int, max_armies : int) -> Dict:

    return Dict(
        {
            "reinforce_move" : MultiDiscrete([game_map.number_of_nodes(), max_armies], start=[0, 1], dtype=np.int16),
            "atk_move" : Dict(
                {
                    "edge" : Discrete(game_map.number_of_edges(), dtype=np.int16),
                    "amount" : Discrete(max_armies, dtype=np.int16),
                    "is_move" : Discrete(2, dtype=np.int8)
                }
            )
        }
    )


def generate_action_mask(game_map : nx.DiGraph, max_armies : int, agent_state : dict, agent_id : int) -> dict:

    # Default no-action allowed mask.
    mask = {
        "reinforce_move" : np.zeros([game_map.number_of_nodes(), max_armies], dtype=np.bool),
        "atk_move" : {
            "edge" : np.zeros([game_map.number_of_edges()], dtype=np.bool),
            "amount" : np.zeros([max_armies], dtype=np.bool),
            "is_move" : np.zeros([2], dtype=np.bool)
        }
    }
    
    if agent_state["is_starting_placement"]:
        # Still placing at the start, can place one in ANY UNOWNED territories.
        mask["reinforce_move"][:, 0] = ~agent_state["owned"]
    elif agent_state["troops_to_place"] > 0:
        # Still has reinforcement troops to place. Must place them in owned territories before attacking.
        owned_nodes = np.where(agent_state["territory_owner"] == agent_id)[0]
        mask["reinforce_move"][owned_nodes, 0:agent_state["troops_to_place"]] = True
    else:
        # No placing required, attacking is allowed.
        owned_nodes = np.where(agent_state["territory_owner"] == agent_id)[0]
        mask["atk_move"] = {
            # TODO: Attacking mask.
        }

    return mask