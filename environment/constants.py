from enum import Enum, IntEnum
from dataclasses import dataclass

class CardType(IntEnum):
    '''
        This class just marks the available card types.
        One can potentially change this class to add custom card types for custom card trades.
    '''
    INFANTRY = 0
    CAVALRY = 1
    ARTILLERY = 2
    JOKER = 3


class RiskPhase(IntEnum):
    '''
        This class is meant to facilitate the conversion between phases and actual applicable numbers inside the observation state.
        Phase description:
            - STARTING_PLACEMENT: Used at the beginning of the game to mark the "picking unowned nodes" phase of the game.
            - SELECT_NODE: Used when reinforcing an already owned territory. Always followed by TROOPS_REINFORCE.
            - SELECT_EDGE: Used when picking an edge for attacking/moving. Followed by either TROOPS_ATTACK or TROOPS_MOVEMENT, depending on the target node.
            - TROOPS_REINFORCE/ATTACK/MOVEMENT: Used when the turn player has to pick the number of troops for the specific move.
            - TRADE_CARDS: Used when the turn player has the chance to trade for cards at the start of their turn.
    '''
    STARTING_PLACEMENT = 0
    SELECT_NODE = 1
    SELECT_EDGE = 2
    TROOPS_REINFORCE = 3
    TROOPS_ATTACK = 4
    TROOPS_MOVEMENT = 5
    TRADE_CARDS = 6


class TradeChoices(IntEnum):
    '''
        This class is meant to allow flexible changes on the possible trade combinations.
        Each element marks the trade that needs to be taken, which is then elaborated in risk._update_state.
    '''
    NO_OP = 0
    TRADE_ARTILLERY = 1
    TRADE_INFANTRY = 2
    TRADE_CAVALRY = 3
    TRADE_MIXED = 4
    TRADE_JOKER = 5


class TroopAction(Enum):
    '''
        This class lists the available actions when it comes to deciding the number of troops to move.
        Values can either be absolute numbers (marked with "abs") or a percentage of the source node's troops to send.
        Note that not all actions may be choosable depending on the masking, but as long as ONE is a choice it's guaranteed to have one viable action.
    '''

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
        obj.kind = kind
        obj.val = val
        obj._value_ = (kind, val)
        return obj

    def to_troops(self, max_amount: int) -> int:
        if self.kind == "abs":
             return min(int(self.val), max_amount)
        return max(1, int(round(max_amount * self.val)))

    @property
    def name_str(self):
        return self.name


@dataclass(frozen=True, slots=True)
class TroopActions:
    _actions: tuple[TroopAction, ...] = tuple(TroopAction)

    def __getitem__(self, idx: int) -> TroopAction:
        return self._actions[idx]

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self):
        yield from self._actions