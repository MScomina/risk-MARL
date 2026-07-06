# https://gymnasium.farama.org/api/spaces/
# Inspiration for environment (masking): https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/chess/chess.py

import networkx as nx
import numpy as np

from dataclasses import dataclass, field
from .constants import (
    CardType,
    RiskPhase,
    TradeChoices,
    TroopAction
)

from .config import CARD_TYPES, CARD_PROBS, TRADE_AMOUNTS, MAX_ATK_DEF_TROOPS

from gymnasium.spaces import Box, Dict, Discrete, Space


# State/observation spaces info:
#   territory_owner: The owner of the territory, -1 if owned by nobody.
#   number_of_armies: Number of armies in a territory, always 0 if unowned.
#   action_phase: Number representing the phase. For more details on each phase check the constants.RiskPhase class.
#   troops_to_place: How many reinforcement troops are left to place (if reinforcement phase).
#   selected_node: Node ID selected in a previous select_node move, -1 if not selected.
#   selected_edge: Edge ID selected in a previous select_edge move, -1 if not selected.

# Action space info:
#   The action is a single number representing, based on the current action_phase, either:
#   - Node ID to pick (if starting placement or reinforcing) + 1 (No-op case)
#   - Edge ID to pick (if selecting edge, usually for attacking or moving) + 1 (No-op case)
#   - Number of armies to use (if attacking, moving or reinforcing, always after picking node/edge)
#   - Card trade combination (if trading cards, usually at the start of the turn)


@dataclass(slots=True)
class RiskHelper():
    '''
        The RiskHelper class is a dataclass used to compute helper functions used in the Risk environment,
        such as computing the initial state, masks for a given state and various game utilities such as 
        battle outcomes and drawing cards.
    '''

    game_map : nx.DiGraph
    num_agents : int
    max_armies : int
    is_card_game : bool
    max_atk_def_troops : tuple[int, int]
    is_blitz : bool

    rng : np.random.Generator = field(default_factory=np.random.default_rng)

    num_nodes: int = field(init=False)
    num_edges: int = field(init=False)
    mask_size: int = field(init=False)

    edge_src: np.ndarray = field(init=False)
    edge_dst: np.ndarray = field(init=False)

    _IS_ABS: np.ndarray = field(init=False)
    _ABS: np.ndarray   = field(init=False)
    _PCT: np.ndarray   = field(init=False)


    # https://docs.python.org/3/library/dataclasses.html#dataclasses.__post_init__
    def __post_init__(self) -> None:
        self.num_nodes, self.num_edges = self.game_map.number_of_nodes(), self.game_map.number_of_edges()
        self.mask_size = max(
            self.num_nodes + 1,
            self.num_edges + 1,
            len(TroopAction),
            len(TradeChoices)
        )

        idx_to_edge = self.game_map.graph["idx_to_edge"]
        node_to_idx = self.game_map.graph["node_to_idx"]

        self.edge_src = np.empty(self.num_edges, dtype=np.int16)
        self.edge_dst = np.empty(self.num_edges, dtype=np.int16)

        for idx, (src, dst) in idx_to_edge.items():
            self.edge_src[idx] = node_to_idx[src]
            self.edge_dst[idx] = node_to_idx[dst]

        self._IS_ABS = np.array([a.kind == "abs" for a in TroopAction])
        self._ABS = np.array([a.val for a in TroopAction], dtype=np.int32)
        self._PCT = np.array([a.val for a in TroopAction], dtype=np.float32)


    def observation_space(self) -> Dict:

        observation_base : dict[str, Space] = {
            "territory_owner": Box(low=-1, high=self.num_agents, shape=(self.num_nodes, ), dtype=np.int8),
            "number_of_armies": Box(low=0, high=self.max_armies+1, shape=(self.num_nodes, ), dtype=np.int16),
            "action_phase" : Discrete(len(RiskPhase), dtype=np.int8),
            "troops_to_place" : Discrete(32767, dtype=np.int16),
            "selected_node": Discrete(self.num_nodes+1, start=-1, dtype=np.int16),
            "selected_edge": Discrete(self.num_edges+1, start=-1, dtype=np.int16),
        }

        if self.is_card_game:
            observation_base["cards_in_hand"] = Box(low=0, high=32767, shape=(len(CardType), ), dtype=np.int16)
            observation_base["amount_cards_others"] = Box(low=0, high=32767, shape=(self.num_agents-1, ), dtype=np.int16)
        
        return Dict(observation_base)


    def action_space(self) -> Dict:

        return Discrete(self.mask_size, dtype=np.int64)


    def mask_space(self) -> Dict:

        return Box(low=0, high=1, shape=(self.mask_size, ), dtype=np.int8)


    def starting_observation(self, full_knowledge : bool = False) -> dict:
        '''
            Generates the starting observations at the start of the game (no owned territories, no troops, no cards, starting placement phase).
        '''

        observation : dict[str, np.ndarray] = {
            "territory_owner" : np.full(shape=self.num_nodes, fill_value=-1, dtype=np.int8),
            "number_of_armies" : np.zeros(self.num_nodes, dtype=np.int16),
            "action_phase" : np.array(0, dtype=np.int8),
            "troops_to_place" : np.array(0, dtype=np.int16),
            "selected_node": np.array(-1, dtype=np.int16),
            "selected_edge": np.array(-1, dtype=np.int16)
        }

        if self.is_card_game:
            if full_knowledge:
                observation["cards_in_hand"] = np.zeros((self.num_agents, len(CardType)), dtype=np.int16)
            else:
                observation["cards_in_hand"] = np.zeros(len(CardType), dtype=np.int16)
                observation["amount_cards_others"] = np.zeros(self.num_agents-1, dtype=np.int16)

        return observation


    def generate_action_mask(self, agent_id : int, agent_state : dict) -> np.ndarray:
        '''
            Given a certain state, computes the action mask of a certain agent_id (the allowed moves on the action space).
            The action mask is a binary array of shape [action_shape, ].
        '''

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

        # No-op action if there's no available nodes.
        # It might happen if an agent has filled all of its nodes or somehow has 0 nodes (should not happen).
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

        # No-op operation if there's no available edges FOR MOVEMENT ACTION.
        # This can happen if an agent's territories are all disjointed, and they would essentially never pass turn without an allowed movement action.
        if not has_movement:
            mask[-1] = 1

        return mask


    def _mask_troops_reinforce(self, state, agent_id):

        assert state["selected_node"] != -1, (
            "Current state is invalid: node has not been selected in troops_reinforce."
        )

        current = state["number_of_armies"][state["selected_node"]]

        max_placeable = min(
            state["troops_to_place"],
            self.max_armies - current,
        )

        return self._valid_troops_mask(state["troops_to_place"], max_placeable)


    def _mask_troops_attack(self, state, agent_id):

        assert state["selected_edge"] != -1, (
            "Current state is invalid: edge has not been selected in troops_attack."
        )

        edge = state["selected_edge"]
        source = state["number_of_armies"][self.edge_src[edge]]

        max_allowed = source - 1
        if not self.is_blitz:
            max_allowed = min(source, self.max_atk_def_troops[0])

        return self._valid_troops_mask(source-1, max_allowed)


    def _mask_troops_movement(self, state, agent_id):
        assert state["selected_edge"] != -1, (
            "Current state is invalid: edge has not been selected in troops_movement."
        )

        edge = state["selected_edge"]
        src = self.edge_src[edge]
        dst = self.edge_dst[edge]

        source = state["number_of_armies"][src]
        dest = state["number_of_armies"][dst]

        movable = min(source - 1, self.max_armies - dest)

        return self._valid_troops_mask(source-1, movable)


    def _mask_trade_cards(self, state : dict, agent_id : int) -> np.ndarray:
        
        mask = np.zeros(len(TradeChoices), dtype=np.int8)

        normal_cards_counts = state["cards_in_hand"][agent_id][[CardType.INFANTRY, CardType.CAVALRY, CardType.ARTILLERY]]

        mask[TradeChoices.NO_OP] = 1
        mask[TradeChoices.TRADE_ARTILLERY] = int(state["cards_in_hand"][agent_id][CardType.ARTILLERY] >= 3)
        mask[TradeChoices.TRADE_INFANTRY] = int(state["cards_in_hand"][agent_id][CardType.INFANTRY] >= 3)
        mask[TradeChoices.TRADE_CAVALRY] = int(state["cards_in_hand"][agent_id][CardType.CAVALRY] >= 3)
        mask[TradeChoices.TRADE_MIXED] = int((normal_cards_counts >= 1).all())
        mask[TradeChoices.TRADE_JOKER] = int(state["cards_in_hand"][agent_id][CardType.JOKER] >= 1 and (normal_cards_counts >= 2).any())

        return mask

    
    def _raw_troops(self, raw: int) -> np.ndarray:
        abs_mask = self._IS_ABS
        pct_vals = np.maximum((raw * self._PCT).astype(int), 1)
        return np.where(abs_mask, self._ABS, pct_vals)


    def _valid_troops_mask(self, raw: int, max_amount: int) -> np.ndarray:
        troops = self._raw_troops(raw)
        return (troops >= 1) & (troops <= max_amount)


    @staticmethod
    def cards_trade_amount(trade_type : int | TradeChoices) -> int:
        '''
            Returns the number of troops received based on the specific trade type.
        '''
        return TRADE_AMOUNTS[trade_type]


    @staticmethod
    def draw_card(rng : np.random.Generator | None = None) -> CardType:
        '''
            Draws a card (returns the CardType, which is an IntEnum, corresponding to the card defined in CARD_TYPES).
        '''
        rng = rng or np.random.default_rng()
        # If one wants to develop an actual drawing setup with a deck, this function is to change.
        # For simplicity, it has just been reduced to drawing from a RNG generator.
        return CardType(
            rng.choice(
                CARD_TYPES,
                p=CARD_PROBS
            )
        )


    @staticmethod
    def risk_attack_outcome(atk_armies : int, def_armies : int, rng: np.random.Generator | None = None, max_atk_def_troops : tuple[int, int] = MAX_ATK_DEF_TROOPS) -> tuple[int, int, int]:
        '''
            Given the number of attack and defense armies (and additional rules such as the number of dices thrown on each side), returns
            the number of remaining armies and the amount of moved armies if the attack side has won.
            NOTE: There is no check for blitz rules in this class since it's always assumed that blitz rules have been checked beforehand,
            through either masking or environment logic.
        '''
        if rng is None:
            rng = np.random.default_rng()

        # Given a number of atk armies and def_armies, reiterate battle until either side is at 0.
        # Battles can be reiterated and have any amount of dices at a time.
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