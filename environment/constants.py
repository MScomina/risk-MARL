from enum import Enum, IntEnum
from dataclasses import dataclass

class CardType(IntEnum):
    INFANTRY = 0
    CAVALRY = 1
    ARTILLERY = 2
    JOKER = 3


class RiskPhase(IntEnum):
    STARTING_PLACEMENT = 0
    SELECT_NODE = 1
    SELECT_EDGE = 2
    TROOPS_REINFORCE = 3
    TROOPS_ATTACK = 4
    TROOPS_MOVEMENT = 5
    TRADE_CARDS = 6


class TradeChoices(IntEnum):
    NO_OP = 0
    TRADE_ARTILLERY = 1
    TRADE_INFANTRY = 2
    TRADE_CAVALRY = 3
    TRADE_MIXED = 4
    TRADE_JOKER = 5


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


@dataclass(frozen=True, slots=True)
class TroopActions:
    _actions: tuple[TroopAction, ...] = tuple(TroopAction)

    def __getitem__(self, idx: int) -> TroopAction:
        return self._actions[idx]

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self):
        yield from self._actions