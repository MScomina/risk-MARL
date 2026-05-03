import networkx as nx
import numpy as np

from gymnasium.spaces import Dict

# State/observation spaces info:
#   owned: Whether a territory is owned or not by someone.
#   territory_owner: The owner of the territory, n_agents if owned by nobody.
#   number_of_armies: Number of armies in a territory.
#   is_starting_placement: Whether it's the starting phase of placing troops on the board or not.
#   troops_to_place: How many reinforcement troops are left to place (if reinforcement phase).

def generic_observation_space(game_map : nx.DiGraph, num_agents : int, max_armies : int) -> Dict:
    
    return Dict(
        {
            "owned": MultiBinary(game_map.number_of_nodes()),
            "territory_owner": MultiDiscrete([num_agents+1] * game_map.number_of_nodes()),
            "number_of_armies": MultiDiscrete([max_armies+1] * game_map.number_of_nodes()),
            "is_starting_placement" : Discrete(2),
            "troops_to_place" : Discrete(self.max_armies)
        }
    )


def generate_starting_observation(game_map : nx.DiGraph, num_agents : int, full_knowledge : bool = False) -> dict:

    return {
        "owned" : np.full(shape=game_map.number_of_nodes(), fill_value=False),
        "territory_owner" : np.full(shape=game_map.number_of_nodes(), fill_value=num_agents),
        "number_of_armies" : np.zeros(game_map.number_of_nodes()),
        "is_starting_placement" : 1,
        "troops_to_place" : 0
    }


def generate_action_mask(map : nx.DiGraph, state : dict):
    pass