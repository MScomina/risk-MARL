import numpy as np
from .constants import CardType, TradeChoices

CLASSIC_MAP_SIZE = 42
DEFAULT_DENSITY = 2.8

# Defines how the DENSE reward's weights are computed.
TERRITORY_REWARD_COEFF = 1.0
TROOPS_REWARD_COEFF = 1.0
CONTINENT_REWARD_COEFF = 5.0

CONTINENT_REWARD_SPIKY = 4.0

# Scales the DENSE reward.
RAW_REWARD_SCALING = 0.5
FINAL_PARTIAL_REWARD_SCALING = 0.2

# Default args.
NUM_AGENTS = 2
MAX_ARMIES = 1_000
MAX_ITERS = 10_000
IS_CARD_GAME = True
DENSE_REWARDS = True
MAX_ATK_DEF_TROOPS = (3, 2)
IS_BLITZ = True

STARTING_REINFORCEMENTS = {
    2: 40,
    3: 35,
    4: 30,
    5: 25,
    6: 20
}

CARD_DRAW_WEIGHTS = {
    CardType.ARTILLERY: 15,
    CardType.INFANTRY:  15,
    CardType.CAVALRY:   10,
    CardType.JOKER:      2,
}

CARD_TYPES = np.array(list(CARD_DRAW_WEIGHTS.keys()), dtype=np.int8)
CARD_PROBS = np.array(
    list(CARD_DRAW_WEIGHTS.values()), dtype=np.float32
) / sum(CARD_DRAW_WEIGHTS.values())

TRADE_AMOUNTS: dict[TradeChoices, int] = {
    TradeChoices.NO_OP:           0,
    TradeChoices.TRADE_ARTILLERY: 4,
    TradeChoices.TRADE_INFANTRY:  6,
    TradeChoices.TRADE_CAVALRY:   8,
    TradeChoices.TRADE_MIXED:     10,
    TradeChoices.TRADE_JOKER:     12,
}