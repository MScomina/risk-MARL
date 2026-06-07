# https://gymnasium.farama.org/api/spaces/
# Inspiration for environment (masking): https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/chess/chess.py

import networkx as nx
import numpy as np

from .maps import graph_utils

import functools
from enum import IntEnum

from pettingzoo import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper

from gymnasium.spaces import Box, Dict, Discrete, Space

# State/observation spaces info:
#   territory_owner: The owner of the territory, -1 if owned by nobody.
#   number_of_armies: Number of armies in a territory.
#   action_phase: Number representing the phase (0: initial placing, 1: select army count, 2: select node, 3: select edge, 4: trade cards)
#   troops_to_place: How many reinforcement troops are left to place (if reinforcement phase).
#   selected_node: Node ID selected in a previous select_node move, -1 if not selected.
#   selected_edge: Edge ID selected in a previous select_edge move, -1 if not selected.

# Action space info:
#   The action is a single number representing, based on the current action_phase, either:
#   - Node ID to pick (if phase 0 or 2, usually meant for reinforcements) + 1 (No-op case)
#   - Edge ID to pick (if phase 3, usually meant for movement/attack) + 1 (No-op case)
#   - Number of armies to use (if phase 1, usually after a node/edge pick move)
#   - Card trade combination (if phase 4, usually at the start of the turn)

class CardTypes(IntEnum):
    INFANTRY = 0
    CAVALRY = 1
    ARTILLERY = 2
    JOKER = 3

class TradeChoices(IntEnum):
    NO_OP = 0
    TRADE_ARTILLERY = 1
    TRADE_INFANTRY = 2
    TRADE_CAVALRY = 3
    TRADE_MIXED = 4
    TRADE_JOKER = 5

class RiskPhase(IntEnum):
    STARTING_PLACEMENT = 0
    SELECT_ARMY_COUNT = 1
    SELECT_NODE = 2
    SELECT_EDGE = 3
    TRADE_CARDS = 4

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


class RiskHelper():

    _TRADE_AMOUNTS : dict[int | TradeChoices, int] = {
        TradeChoices.NO_OP: 0,
        TradeChoices.TRADE_ARTILLERY: 4,
        TradeChoices.TRADE_INFANTRY: 6,
        TradeChoices.TRADE_CAVALRY: 8,
        TradeChoices.TRADE_MIXED: 10,
        TradeChoices.TRADE_JOKER: 12
    }

    _CARD_DRAW_WEIGHTS : dict[int | CardTypes, int] = {
        CardTypes.ARTILLERY: 15,
        CardTypes.INFANTRY: 15,
        CardTypes.CAVALRY: 10,
        CardTypes.JOKER: 2
    }

    _CARD_TYPES = np.array(
        list(_CARD_DRAW_WEIGHTS.keys()),
        dtype=np.int8
    )

    _CARD_PROBS = np.array(
        list(_CARD_DRAW_WEIGHTS.values()),
        dtype=np.float32
    )

    _CARD_PROBS /= _CARD_PROBS.sum()

    def __init__(self, game_map : nx.DiGraph, num_agents : int, max_armies : int, is_card_game : bool):

        self.game_map = game_map
        self.num_agents = num_agents
        self.max_armies = max_armies

        self.num_nodes = self.game_map.number_of_nodes()
        self.num_edges = self.game_map.number_of_edges()
        self.mask_size = max(self.num_nodes + 1, self.num_edges + 1, self.max_armies, len(TradeChoices))

        self.is_card_game = is_card_game

        self.edge_src = np.empty(self.num_edges, dtype=np.int16)
        self.edge_dst = np.empty(self.num_edges, dtype=np.int16)

        for idx, (src, dst) in self.game_map.graph["idx_to_edge"].items():
            self.edge_src[idx] = self.game_map.graph["node_to_idx"][src]
            self.edge_dst[idx] = self.game_map.graph["node_to_idx"][dst]

    def observation_space(self) -> Dict:

        observation_base = {
            "territory_owner": Box(low=-1, high=self.num_agents, shape=(self.num_nodes, ), dtype=np.int8),
            "number_of_armies": Box(low=0, high=self.max_armies+1, shape=(self.num_nodes, ), dtype=np.int16),
            "action_phase" : Discrete(len(RiskPhase), dtype=np.int8),
            "troops_to_place" : Discrete(self.max_armies, dtype=np.int16),
            "selected_node": Discrete(self.num_nodes+1, start=-1, dtype=np.int16),
            "selected_edge": Discrete(self.num_edges+1, start=-1, dtype=np.int16),
        }

        if self.is_card_game:
            observation_base["cards_in_hand"] = Box(low=0, high=32767, shape=(len(CardTypes), ), dtype=np.int16)
            observation_base["amount_cards_others"] = Box(low=0, high=32767, shape=(self.num_agents-1, ), dtype=np.int16)
        
        return Dict(observation_base)

    def starting_observation(self, full_knowledge : bool = False) -> dict:

        observation = {
            "territory_owner" : np.full(shape=self.num_nodes, fill_value=-1, dtype=np.int8),
            "number_of_armies" : np.zeros(self.num_nodes, dtype=np.int16),
            "action_phase" : np.array(0, dtype=np.int8),
            "troops_to_place" : np.array(0, dtype=np.int16),
            "selected_node": np.array(-1, dtype=np.int16),
            "selected_edge": np.array(-1, dtype=np.int16)
        }

        if self.is_card_game:
            if full_knowledge:
                observation["cards_in_hand"] = np.zeros((self.num_agents, len(CardTypes)), dtype=np.int16)
            else:
                observation["cards_in_hand"] = np.zeros(len(CardTypes), dtype=np.int16)
                observation["amount_cards_others"] = np.zeros(self.num_agents-1, dtype=np.int16)

        return observation

    def action_space(self) -> Dict:

        return Discrete(self.mask_size, dtype=np.int64)

    def mask_space(self) -> Dict:

        return Box(low=0, high=1, shape=(self.mask_size, ), dtype=np.int8)

    def generate_action_mask(self, agent_id : int, agent_state : dict) -> np.ndarray:

        match agent_state["action_phase"]:

            case RiskPhase.STARTING_PLACEMENT:
                mask = self._mask_starting_placement(
                    state=agent_state
                )

            case RiskPhase.SELECT_NODE:
                mask = self._mask_select_node(
                    state=agent_state,
                    agent_id=agent_id
                )

            case RiskPhase.SELECT_EDGE:
                mask = self._mask_select_edge(
                    state=agent_state, 
                    agent_id=agent_id
                )

            case RiskPhase.SELECT_ARMY_COUNT:
                mask = self._mask_select_army_count(
                    state=agent_state,
                    agent_id=agent_id
                )

            case RiskPhase.TRADE_CARDS:
                if not self.is_card_game:
                    raise ValueError(f"Phase {agent_state['action_phase']} in use, despite not having is_card_game to True.")
                mask = self._mask_trade_cards(
                    state=agent_state,
                    agent_id=agent_id
                )

            case _:
                raise ValueError(f"Undefined phase: {agent_state['action_phase']}")

        full_mask = np.zeros([self.mask_size], dtype=np.int8)
        full_mask[:mask.size] = mask

        return full_mask

    
    def _mask_starting_placement(self, state : dict) -> np.ndarray:
        # Still placing at the start, can place one in ANY UNOWNED territories.
        return (state["territory_owner"] == -1).astype(np.int8)

    def _mask_select_node(self, agent_id : int, state : dict) -> np.ndarray:
        # Since the only time one can pick a node that is NOT the starting placement phase is to reinforce, one can pick any owned nodes.
        mask = np.zeros(self.num_nodes+1, dtype=np.int8)

        owned = (state["territory_owner"] == agent_id)
        can_expand = (state["number_of_armies"] < self.max_armies)
        valid_nodes = owned & can_expand
        
        mask[:-1] = valid_nodes

        if not valid_nodes.any():
            mask[-1] = 1

        return mask

    def _mask_select_edge(self, state: dict, agent_id: int) -> np.ndarray:

        mask = np.zeros(self.num_edges + 1, dtype=np.int8)

        owners = state["territory_owner"]
        armies = state["number_of_armies"]

        src_owned = owners[self.edge_src] == agent_id
        src_can_act = armies[self.edge_src] > 1

        valid_src = src_owned & src_can_act

        dst_owned = owners[self.edge_dst] == agent_id
        dst_full = armies[self.edge_dst] >= self.max_armies

        mask[:-1] = valid_src & ~(dst_owned & dst_full)

        has_movement = np.any(
            valid_src &
            dst_owned &
            ~dst_full
        )

        if not has_movement:
            mask[-1] = 1

        return mask

    def _mask_select_army_count(self, state : dict, agent_id : int) -> np.ndarray:

        mask = np.zeros(self.max_armies, dtype=np.int8)
        owners = state["territory_owner"]

        if state["troops_to_place"] > 0:
            # Still has reinforcement troops to place. Assuming a node has already been picked.
            if state["selected_node"] == -1:
                raise RuntimeError(f"Inconsistent world space: selected_node is -1.")
            current_troops = state["number_of_armies"][state["selected_node"]]
            mask[1:min(state["troops_to_place"]+1,self.max_armies-current_troops+1)] = 1
        else:
            # No reinforcement troops means this is likely an attack/move action.
            if state["selected_edge"] == -1:
                raise RuntimeError(f"Inconsistent world space: selected_edge is -1.")
            edge = self.game_map.graph["idx_to_edge"][state["selected_edge"]]
            source_node_armies = state["number_of_armies"][self.game_map.graph["node_to_idx"][edge[0]]]
            dst = self.game_map.graph["node_to_idx"][self.game_map.graph["idx_to_edge"][state["selected_edge"]][1]]
            dst_node_armies = state["number_of_armies"][dst]
            if owners[dst] == agent_id:
                # This is a movement action.
                mask[1:min(source_node_armies, self.max_armies-dst_node_armies+1)] = 1
            else:
                # This is an attack action.
                mask[1:min(source_node_armies, 4)] = 1

        return mask

    def _mask_trade_cards(self, state : dict, agent_id : int) -> np.ndarray:
        
        mask = np.zeros(len(TradeChoices), dtype=np.int8)

        normal_cards_counts = state["cards_in_hand"][agent_id][[CardTypes.INFANTRY, CardTypes.CAVALRY, CardTypes.ARTILLERY]]

        mask[TradeChoices.NO_OP] = 1
        mask[TradeChoices.TRADE_ARTILLERY] = int(state["cards_in_hand"][agent_id][CardTypes.ARTILLERY] >= 3)
        mask[TradeChoices.TRADE_INFANTRY] = int(state["cards_in_hand"][agent_id][CardTypes.INFANTRY] >= 3)
        mask[TradeChoices.TRADE_CAVALRY] = int(state["cards_in_hand"][agent_id][CardTypes.CAVALRY] >= 3)
        mask[TradeChoices.TRADE_MIXED] = int((normal_cards_counts >= 1).all())
        mask[TradeChoices.TRADE_JOKER] = int(state["cards_in_hand"][agent_id][CardTypes.JOKER] >= 1 and (normal_cards_counts >= 2).any())

        return mask

    def _generate_continent_masks(self) -> dict:
        
        continent_masks = {}
        for continent, data in self.game_map.graph["continents"].items():
            mask = np.zeros(self.num_nodes, dtype=bool)
            for terr in data["territories"]:
                idx = self.game_map.graph["node_to_idx"][terr]
                mask[idx] = True
            continent_masks[continent] = mask

        return continent_masks

    @staticmethod
    def cards_trade_amount(trade_type : int | TradeChoices) -> int:
        return RiskHelper._TRADE_AMOUNTS[trade_type]

    @staticmethod
    def draw_card(rng : np.random.Generator | None = None) -> int:
        if rng is None:
            rng = np.random.default_rng()
        # If one wants to develop an actual drawing setup with a deck, this function is to change.
        # For simplicity, it has just been reduced to drawing from a RNG generator.
        return rng.choice(
            RiskHelper._CARD_TYPES,
            p=RiskHelper._CARD_PROBS
        )


    @staticmethod
    def risk_attack_outcome(atk_armies : int, def_armies : int, rng: np.random.Generator | None = None) -> tuple[int, int]:
        if rng is None:
            rng = np.random.default_rng()
        # Given a number of attackers and defenders, returns the number of losses on both sides.
        atk_dice = min(3, atk_armies)
        def_dice = min(2, def_armies)

        atk_roll = rng.integers(1, 7, size=atk_dice)
        def_roll = rng.integers(1, 7, size=def_dice)

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