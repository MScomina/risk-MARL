import networkx as nx
import numpy as np

from maps import graph_utils

from gymnasium.spaces import Dict, Discrete, MultiDiscrete, MultiBinary

# State/observation spaces info:
#   owned: Whether a territory is owned or not by someone.
#   territory_owner: The owner of the territory, -1 if owned by nobody.
#   number_of_armies: Number of armies in a territory.
#   is_starting_placement: Whether it's the starting phase of placing troops on the board or not.
#   troops_to_place: How many reinforcement troops are left to place (if reinforcement phase).

# Action space info:
#   reinforce_move: Reinforce move (place troops on node).
#   atk_move: Directed edge of graph where the movement/attack happens, number of armies and
#             whether it's a movement or attack (0=movement, 1=attack).

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
        "owned" : np.full(shape=game_map.number_of_nodes(), fill_value=False, dtype=bool),
        "territory_owner" : np.full(shape=game_map.number_of_nodes(), fill_value=-1, dtype=np.int8),
        "number_of_armies" : np.zeros(game_map.number_of_nodes(), dtype=np.int16),
        "is_starting_placement" : 1,
        "troops_to_place" : 0
    }


def generic_action_space(game_map : nx.DiGraph, num_agents : int, max_armies : int) -> Dict:

    return Dict(
        {
            "reinforce_move" : MultiDiscrete([game_map.number_of_nodes(), max_armies], start=[0, 1], dtype=np.int16),
            "atk_move" : MultiDiscrete([game_map.number_of_edges(), max_armies, 2], start=[0,1,0], dtype=np.int16)
        }
    )


def generate_action_mask(game_map : nx.DiGraph, max_armies : int, agent_state : dict, agent_id : int) -> dict:

    # Default no-action allowed mask.
    mask = {
        "reinforce_move" : np.zeros([game_map.number_of_nodes(), max_armies], dtype=bool),
        "atk_move" : np.zeros([game_map.number_of_edges(), max_armies, 2], dtype=bool)
    }
    
    if agent_state["is_starting_placement"]:
        # Still placing at the start, can place one in ANY UNOWNED territories.
        mask["reinforce_move"][:, 0] = ~agent_state["owned"]
    elif agent_state["troops_to_place"] > 0:
        # Still has reinforcement troops to place. Must place them in owned territories before attacking.
        owned_nodes = np.where(agent_state["territory_owner"] == agent_id)[0]
        mask["reinforce_move"][owned_nodes, 0:agent_state["troops_to_place"]] = True
    else:
        # No placing required, attacking/moving is allowed.
        owned_nodes = set(np.where(agent_state["territory_owner"] == agent_id)[0].tolist())
        for node in owned_nodes:
            for edge in game_map.out_edges(game_map.graph["idx_to_node"][node]):
                if game_map.graph["node_to_idx"][edge[1]] in owned_nodes:
                    # Target node is owned by the same person, only movement allowed.
                    mask["atk_move"][game_map.graph["edge_to_idx"][edge], :agent_state["number_of_armies"][node]-1, 0] = True
                else:
                    # Target node is owned by somebody else, only attack allowed.
                    mask["atk_move"][game_map.graph["edge_to_idx"][edge], :min(agent_state["number_of_armies"][node]-1,3), 1] = True

    return mask


def risk_attack_outcome(atk_armies : int, def_armies : int) -> tuple[int, int]:
    # Given a number of attackers and defenders, returns the number of losses on both sides.
    atk_dice_amount = min(3, atk_armies)
    def_dice_amount = min(2, def_armies)

    atk_roll = np.random.randint(1, 7, size=atk_dice)
    def_roll = np.random.randint(1, 7, size=def_dice)

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