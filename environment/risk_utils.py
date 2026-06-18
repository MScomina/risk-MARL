# https://gymnasium.farama.org/api/spaces/
# Inspiration for environment (masking): https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/chess/chess.py

import networkx as nx
import numpy as np

from .maps import graph_utils

import functools
from dataclasses import dataclass
from enum import Enum, IntEnum

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
#   - Node ID to pick (if phase 0 or 1, usually meant for reinforcements) + 1 (No-op case)
#   - Edge ID to pick (if phase 2, usually meant for movement/attack) + 1 (No-op case)
#   - Number of armies to use (if phase 3, 4 or 5, usually after a node/edge pick move)
#   - Card trade combination (if phase 6, usually at the start of the turn)

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
    SELECT_NODE = 1
    SELECT_EDGE = 2
    TROOPS_REINFORCE = 3
    TROOPS_ATTACK = 4
    TROOPS_MOVEMENT = 5
    TRADE_CARDS = 6



class TroopAction(Enum):

    ONE = ("abs", 1)
    TWO = ("abs", 2)
    THREE = ("abs", 3)
    P1  = ("pct", 0.01)
    P3  = ("pct", 0.03)
    P5  = ("pct", 0.05)
    P10 = ("pct", 0.10)
    P15 = ("pct", 0.15)
    P20 = ("pct", 0.20)
    P30 = ("pct", 0.30)
    P40 = ("pct", 0.40)
    P50 = ("pct", 0.50)
    P60 = ("pct", 0.60)
    P80 = ("pct", 0.80)
    P100 = ("pct", 1.00)

    def __new__(cls, kind: str, val: float):
        obj = object.__new__(cls)
        obj._value_ = val
        obj.kind = kind
        return obj

    def to_troops(self, max_amount: int) -> int:
        if self.kind == "abs":
             return min(int(self.value), max_amount)
        return max(1, int(round(max_amount * self.value)))

    @property
    def name_str(self):
        return self.name




class TroopActionsMeta(type):
    def __getitem__(cls, index):
        return cls._TROOP_ACTIONS[index]

    def __len__(cls) -> int:
        return len(cls._TROOP_ACTIONS)

    def __iter__(cls):
        return iter(cls._TROOP_ACTIONS)




class TroopActions(metaclass=TroopActionsMeta):
    _TROOP_ACTIONS = tuple(TroopAction)


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

    IS_ABS = np.array([a.kind == "abs" for a in TroopActions])
    ABS = np.array([a.value for a in TroopActions])
    PCT = np.array([a.value for a in TroopActions])

    _CARD_PROBS /= _CARD_PROBS.sum()


    def __init__(self, game_map : nx.DiGraph, num_agents : int, max_armies : int, is_card_game : bool, max_atk_def_troops : tuple[int, int], 
                 is_blitz : bool, rng: np.random.Generator | None = None):

        self.game_map = game_map
        self.num_agents = num_agents
        self.max_armies = max_armies

        self.num_nodes = self.game_map.number_of_nodes()
        self.num_edges = self.game_map.number_of_edges()
        self.mask_size = max(self.num_nodes + 1, self.num_edges + 1, len(TroopActions), len(TradeChoices))

        self.is_card_game = is_card_game

        self.max_atk_def_troops = max_atk_def_troops
        self.is_blitz = is_blitz

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

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
            "troops_to_place" : Discrete(32767, dtype=np.int16),
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

            case RiskPhase.TROOPS_REINFORCE:
                mask = self._mask_troops_reinforce(
                    state=agent_state,
                    agent_id=agent_id
                )

            case RiskPhase.TROOPS_ATTACK:
                mask = self._mask_troops_attack(
                    state=agent_state,
                    agent_id=agent_id
                )

            case RiskPhase.TROOPS_MOVEMENT:
                mask = self._mask_troops_movement(
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


    def _mask_troops_reinforce(self, state, agent_id):
        mask = np.zeros(len(TroopActions), dtype=np.int8)

        assert state["selected_node"] != -1, (
            "Current state is invalid: node has not been selected in troops_reinforce."
        )

        current = state["number_of_armies"][state["selected_node"]]

        max_placeable = min(
            state["troops_to_place"],
            self.max_armies - current,
        )

        mask[:] = self._valid_troops_mask(state["troops_to_place"], max_placeable)
        return mask


    def _mask_troops_attack(self, state, agent_id):
        mask = np.zeros(len(TroopActions), dtype=np.int8)

        assert state["selected_edge"] != -1, (
            "Current state is invalid: edge has not been selected in troops_attack."
        )

        edge = state["selected_edge"]
        source = state["number_of_armies"][self.edge_src[edge]]

        max_allowed = source - 1
        if not self.is_blitz:
            max_allowed = min(source, self.max_atk_def_troops[0])

        mask[:] = self._valid_troops_mask(source-1, max_allowed)
        return mask


    def _mask_troops_movement(self, state, agent_id):
        mask = np.zeros(len(TroopActions), dtype=np.int8)

        assert state["selected_edge"] != -1, (
            "Current state is invalid: edge has not been selected in troops_movement."
        )

        edge = state["selected_edge"]
        src = self.edge_src[edge]
        dst = self.edge_dst[edge]

        source = state["number_of_armies"][src]
        dest = state["number_of_armies"][dst]

        movable = min(source - 1, self.max_armies - dest)

        mask[:] = self._valid_troops_mask(source-1, movable)
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
    def risk_attack_outcome(atk_armies : int, def_armies : int, rng: np.random.Generator | None = None, max_atk_def_troops : tuple[int, int] = (3, 2)) -> tuple[int, int]:
        
        if rng is None:
            rng = np.random.default_rng()

        # Given a number of atk armies and def_armies, reiterate battle until either side is at 0.
        # Battles can be reiterated and have any amount of dices at a time.
        # No is_blitz check is here because masking takes care of the max amount of armies anyways.
        max_a, max_d = max_atk_def_troops
        starting_def = def_armies

        total_atk_losses = 0
        total_def_losses = 0
        last_atk_dice = 0

        while atk_armies > 0 and def_armies > 0:

            atk_dice = min(max_a, atk_armies)
            def_dice = min(max_d, def_armies)

            last_atk_dice = atk_dice

            k = min(atk_dice, def_dice)

            atk = rng.integers(1, 7, size=atk_dice)
            df  = rng.integers(1, 7, size=def_dice)

            atk_top = np.partition(atk, -k)[-k:]
            def_top = np.partition(df, -k)[-k:]

            atk_top.sort()
            def_top.sort()

            losses_def = np.sum(atk_top > def_top)
            losses_att = k - losses_def

            atk_armies -= losses_att
            def_armies -= losses_def

            total_atk_losses += losses_att
            total_def_losses += losses_def

        if total_def_losses != starting_def:
            # Defense still has troops, attack has lost. No troops to move.
            return total_atk_losses, total_def_losses, 0

        # Attack has won.
        return total_atk_losses, total_def_losses, int(last_atk_dice)

    def _raw_troops(self, max_amount: int):
        abs_troops = self.ABS
        pct_troops = np.maximum(1, (max_amount * self.PCT).astype(int))

        return np.where(self.IS_ABS, abs_troops, pct_troops)

    def _valid_troops_mask(self, raw_troops : int, max_amount: int):
        troops = self._raw_troops(raw_troops)

        return (
            (troops >= 1) &
            (troops <= max_amount)
        )