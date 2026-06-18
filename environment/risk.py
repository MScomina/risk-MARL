# https://gymnasium.farama.org/api/env/
# https://pettingzoo.farama.org/content/environment_creation/
# https://arxiv.org/pdf/2402.07411 (Potential-Based Reward Shaping For Intrinsic Motivation)

from copy import copy
import functools

import numpy as np

from gymnasium.utils import seeding, EzPickle
from gymnasium.spaces import Dict
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from pathlib import Path

from .maps import graph_utils
from .risk_utils import (
    CardTypes, 
    FlattenObservationWrapper, 
    RiskPhase, 
    RiskHelper, 
    TradeChoices,
    TroopActions
)

# Environment definitions:
#   (Underlying) Space states:
#       - Ownership of territories (part of observation)
#       - Amount of armies in each territory (part of observation)
#       - Action phase (0: initial placing, 1: select army count, 2: select node, 3: select edge, 4: select card trade)
#       - Cards in hand (partial observability of each agent)
#   Actions:
#       The actions are defined based on the state, they could represent, based on the phase:
#       - Node ID to pick
#       - Edge ID to pick
#       - Number of armies to use
#       - Type of card trade to perform

NUM_AGENTS = 2
MAX_ARMIES = 1_000
MAX_ITERS = 10_000
IS_CARD_GAME = True
DENSE_REWARDS = True
MAX_ATK_DEF_TROOPS = (3, 2)
IS_BLITZ = True

def env(should_flatten_obs : bool = False, **kwargs) -> AECEnv:
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    if should_flatten_obs:
        env = FlattenObservationWrapper(env)
    return env


class raw_env(AECEnv, EzPickle):

    metadata : dict = {"render_modes": ["human"], "name": "risk-v1"}

    _STARTING_REINFORCEMENTS = {
        2: 40,
        3: 35,
        4: 30,
        5: 25,
        6: 20
    }

    def __init__(self, render_mode=None, n_agents : int = NUM_AGENTS, map_path : Path | None = None,
                 max_armies : int = MAX_ARMIES, max_iters : int = MAX_ITERS, is_card_game : bool = IS_CARD_GAME,
                 dense_rewards : bool = DENSE_REWARDS, max_atk_def_troops : tuple[int, int] = MAX_ATK_DEF_TROOPS,
                 is_blitz : bool = IS_BLITZ, rng: np.random.Generator | None = None, gamma : float = 0.999):

        EzPickle.__init__(self, render_mode, n_agents, map_path, max_armies, max_iters, is_card_game)
        super().__init__()

        self.agents = ["player_" + str(r) for r in range(n_agents)]
        self.possible_agents = self.agents[:]
        self.n_agents = n_agents
        self.map_network = graph_utils.generate_graph(map_path)
        self.max_armies = max_armies
        self.max_iters = max_iters
        self.is_card_game = is_card_game
        self.num_nodes = self.map_network.number_of_nodes()
        self.num_edges = self.map_network.number_of_edges()
        self.log_max_troops = np.log1p(self.max_armies * self.num_nodes)
        self.dense_rewards = dense_rewards
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        # The idea with is_blitz is that one can commit more troops and the environment will 
        # automatically resolve the battle with that many troops over and over.
        # False: Only singular attack instances are allowed
        # True: agents can send any amount they have, the battles will go until armies on either end finish.
        self.is_blitz = is_blitz
        self.max_atk_def_troops = max_atk_def_troops
        self.gamma = gamma

        self.risk_helper = RiskHelper(
            game_map=self.map_network,
            num_agents=self.n_agents,
            max_armies=self.max_armies,
            is_card_game=self.is_card_game,
            max_atk_def_troops=self.max_atk_def_troops,
            is_blitz=self.is_blitz,
            rng=self.rng
        )

        self._observation_spaces = {
            agent: Dict(
                {
                    "observation" : self.risk_helper.observation_space(),
                    "action_mask" : self.risk_helper.mask_space()
                }
            ) for agent in self.possible_agents
        }
        self._action_spaces = Dict(
            {
                agent: self.risk_helper.action_space() for agent in self.possible_agents
            }
        )

        continent_masks = self.risk_helper._generate_continent_masks()
        self.max_continent_bonus = sum(
            c["bonus"] for c in self.map_network.graph["continents"].values()
        )
        self.continent_data = [
            (continent_masks[name], data["bonus"])
            for name, data in self.map_network.graph["continents"].items()
        ]

        self._continent_lookup = {}
        self._continent_reward_spiky = 2.0

        for mask, _ in self.continent_data:
            n = np.count_nonzero(mask)

            if n not in self._continent_lookup:
                p = np.arange(n + 1) / n
                self._continent_lookup[n] = (
                    np.exp(self._continent_reward_spiky * p) - 1
                ) / (
                    np.exp(self._continent_reward_spiky) - 1
                )

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.current_agent_idx = 0
        self.strength_dirty = True
        self.has_received_starting_reinforcement = np.zeros(self.n_agents, dtype=bool)

        self.render_mode = render_mode


    @functools.lru_cache(maxsize=1000)
    def observation_space(self, agent):
        return self._observation_spaces[agent]


    @functools.lru_cache(maxsize=1000)
    def action_space(self, agent):
        return self._action_spaces[agent]


    def reset(self, seed=None, options=None):

        self.timestep = 0

        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.risk_helper.rng = self.rng

        self.agents = copy(self.possible_agents)
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self._clear_rewards()
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.agent_to_idx = {
            agent: i
            for i, agent in enumerate(self.agents)
        }

        self.world_state = self.risk_helper.starting_observation(full_knowledge=True)
    
        self.observations = {
            agent : self._compute_observation(agent) for agent in self.agents
        }

        self.num_moves = 0

        self.has_conquered_this_turn = False
        self.is_first_turn = True
        self.has_received_starting_reinforcement[:] = False

        self.current_agent_idx = self.agent_to_idx[self.agent_selection]
        self.prev_phi = np.zeros(self.n_agents)
        self.reward_ema = np.zeros(self.n_agents, dtype=np.float32)
        self.territory_counts = np.zeros(self.n_agents, dtype=np.int16)
        self.troop_counts = np.zeros(self.n_agents, dtype=np.int32)
        self.strength_dirty = True


    def observe(self, agent: str):
        self.observations[agent] = self._compute_observation(agent)
        return self.observations[agent]


    def close(self):
        pass


    def step(self, action):

        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        current_agent = self.agent_selection
        self.current_agent_idx = self.agent_to_idx[current_agent]

        self._cumulative_rewards[current_agent] = 0
        self._clear_rewards()

        should_end_turn = self._update_state(current_agent, action)
        if self.dense_rewards and self.strength_dirty:
            self._update_strength()

        if self.territory_counts[self.current_agent_idx] == self.num_nodes:
            print(f"{current_agent} has won!")
            for agent in self.agents:
                if agent == current_agent:
                    self.rewards[agent] = 1.0
                else:
                    self.rewards[agent] = -1.0
                self.terminations[agent] = True
        elif should_end_turn:
            next_agent = self._agent_selector.next()
            while (self.terminations[next_agent] or self.truncations[next_agent]) and next_agent != current_agent:
                if all(self.terminations[a] or self.truncations[a] for a in self.agents):
                    break
                next_agent = self._agent_selector.next()

            self.agent_selection = next_agent
            self.current_agent_idx = self.agent_to_idx[self.agent_selection]
            if not self.world_state["action_phase"] == RiskPhase.STARTING_PLACEMENT:
                self.world_state["troops_to_place"] = self._compute_reinforcements()
            self.has_conquered_this_turn = False

        self.num_moves += 1
        if self.num_moves >= self.max_iters:
            if self.strength_dirty:
                self._update_strength()

            final_strength = self.prev_phi.copy()

            for i, agent in enumerate(self.agents):

                self.rewards[agent] += 0.2 * float(final_strength[i])**2

                self.truncations[agent] = True

        self._accumulate_rewards()


    def _update_state(self, agent : str, action : dict) -> bool:

        # Updates the state and returns whether the turn is over or not.
        match self.world_state["action_phase"]:

            case RiskPhase.STARTING_PLACEMENT:
                if self.world_state["territory_owner"][action] != -1:
                    raise RuntimeError(f"Agent {agent} tried to own an already owned territory: {action}")
                self.world_state["territory_owner"][action] = self.current_agent_idx
                self.world_state["number_of_armies"][action] = 1
                self.territory_counts[self.current_agent_idx] += 1
                self.troop_counts[self.current_agent_idx] += 1
                self.strength_dirty = True
                if np.all(self.world_state["territory_owner"] != -1):
                    # All nodes have been taken, reinforcement has begun.
                    self.world_state["action_phase"] = RiskPhase.SELECT_NODE
                return True

            case RiskPhase.SELECT_NODE:
                if action == self.num_nodes:
                    # No-op operation, either because agent is dead or all nodes are filled. Skip to next phase.
                    self.world_state["troops_to_place"] = 0
                    self.world_state["action_phase"] = RiskPhase.SELECT_EDGE
                    return False
                if action > self.num_nodes:
                    raise RuntimeError(f"Agent {agent} tried to pick a node index greater than the number of nodes: {action}")
                if self.world_state["territory_owner"][action] != self.current_agent_idx:
                    raise RuntimeError(f"Agent {agent} tried to pick another player's territory for action: {action}")
                self.world_state["selected_node"] = action
                self.world_state["action_phase"] = RiskPhase.TROOPS_REINFORCE
                return False

            case RiskPhase.SELECT_EDGE:
                if action == self.num_edges:
                    # No-op operation (no actions available, must pass turn)
                    if self.is_card_game:
                        self.world_state["action_phase"] = RiskPhase.TRADE_CARDS
                    else:
                        self.world_state["action_phase"] = RiskPhase.SELECT_NODE
                    self.world_state["selected_edge"] = -1
                    return True
                if action > self.num_edges:
                    raise RuntimeError(f"Agent {agent} tried to pick an edge index greater than the number of edges: {action}")

                edge_id = action
                src_node = self.risk_helper.edge_src[edge_id]
                dst_node = self.risk_helper.edge_dst[edge_id]

                if self.world_state["territory_owner"][src_node] != self.current_agent_idx:
                    raise RuntimeError(f"Agent {agent} tried to pick an invalid edge ({src_node} is owned by {self.world_state['territory_owner'][src_node]}): {action}")
                self.world_state["selected_edge"] = action
                if self.world_state["territory_owner"][dst_node] != self.current_agent_idx:
                    self.world_state["action_phase"] = RiskPhase.TROOPS_ATTACK
                else:
                    self.world_state["action_phase"] = RiskPhase.TROOPS_MOVEMENT
                return False

            case RiskPhase.TROOPS_REINFORCE:
                # Reinforcing own territory
                troop_storage = self.world_state["troops_to_place"]
                troop_reinforce_amount = TroopActions[action].to_troops(troop_storage)
                self.world_state["number_of_armies"][self.world_state["selected_node"]] += troop_reinforce_amount
                self.troop_counts[self.current_agent_idx] += troop_reinforce_amount
                self.world_state["troops_to_place"] -= troop_reinforce_amount
                self.strength_dirty = True
                if self.world_state["troops_to_place"] > 0:
                    self.world_state["action_phase"] = RiskPhase.SELECT_NODE     
                    self.world_state["selected_node"] = -1
                    return False
                else:
                    if self.is_first_turn:
                        # Placed all starting troops, but others still have to place their starting ones.
                        self.world_state["action_phase"] = RiskPhase.SELECT_NODE     
                        self.world_state["selected_node"] = -1
                        return True
                    self.world_state["action_phase"] = RiskPhase.SELECT_EDGE
                self.world_state["selected_node"] = -1
                return False

            case RiskPhase.TROOPS_ATTACK:
                # Attacking.
                edge_id = self.world_state["selected_edge"]
                src_node = self.risk_helper.edge_src[edge_id]
                dst_node = self.risk_helper.edge_dst[edge_id]

                max_troops = self.world_state["number_of_armies"][src_node]-1
                if not self.is_blitz:
                    max_troops = min(self.max_atk_def_troops[0],self.world_state["number_of_armies"][src_node]-1)

                troop_attack_amount = TroopActions[action].to_troops(max_troops)
                previous_owner = self.world_state["territory_owner"][dst_node]

                dst_amount = self.world_state["number_of_armies"][dst_node]
                atk_losses, def_losses, atk_last_amount = self.risk_helper.risk_attack_outcome(troop_attack_amount, dst_amount, max_atk_def_troops=self.max_atk_def_troops, rng=self.rng)

                self.world_state["number_of_armies"][src_node] -= atk_losses
                self.world_state["number_of_armies"][dst_node] -= def_losses

                self.troop_counts[self.current_agent_idx] -= atk_losses
                self.troop_counts[previous_owner] -= def_losses

                if self.world_state["number_of_armies"][dst_node] == 0:
                    # No troops left, control switches.
                    if not self.has_conquered_this_turn and self.is_card_game:
                        self.world_state["cards_in_hand"][self.current_agent_idx][self.risk_helper.draw_card(rng=self.rng)] += 1
                        self.has_conquered_this_turn = True
                    self.territory_counts[self.current_agent_idx] += 1
                    self.territory_counts[previous_owner] -= 1
                    self.world_state["territory_owner"][dst_node] = self.current_agent_idx
                    self.world_state["number_of_armies"][dst_node] += atk_last_amount
                    self.world_state["number_of_armies"][src_node] -= atk_last_amount
                    if self.territory_counts[previous_owner] <= 0:
                        self.terminations[self.agents[previous_owner]] = True
                        if self.is_card_game:
                            self.world_state["cards_in_hand"][self.current_agent_idx] += self.world_state["cards_in_hand"][previous_owner]
                            self.world_state["cards_in_hand"][previous_owner] = np.zeros(len(CardTypes), dtype=np.int16)
                        if self.dense_rewards:
                            self.rewards[agent] += 0.3 / (self.num_agents-1)    # Wiping opponent is good
                            self.rewards[self.agents[previous_owner]] -= 1.0    # Dying is bad

                self.world_state["action_phase"] = RiskPhase.SELECT_EDGE
                self.world_state["selected_edge"] = -1
                self.strength_dirty = True
                return False

            case RiskPhase.TROOPS_MOVEMENT:
                # Moving.
                edge_id = self.world_state["selected_edge"]
                src_node = self.risk_helper.edge_src[edge_id]
                dst_node = self.risk_helper.edge_dst[edge_id]
                moveable_troops = self.world_state["number_of_armies"][src_node]-1
                troops_movement_amount = TroopActions[action].to_troops(moveable_troops)

                self.world_state["number_of_armies"][src_node] -= troops_movement_amount
                self.world_state["number_of_armies"][dst_node] += troops_movement_amount
                if self.is_card_game:
                    self.world_state["action_phase"] = RiskPhase.TRADE_CARDS
                else:
                    self.world_state["action_phase"] = RiskPhase.SELECT_NODE
                self.world_state["selected_edge"] = -1
                self.strength_dirty = True
                return True

            case RiskPhase.TRADE_CARDS:
                self.world_state["troops_to_place"] += self.risk_helper.cards_trade_amount(action)
                self._update_cards(action=action)
                self.world_state["action_phase"] = RiskPhase.SELECT_NODE
                return False

            case _:
                raise ValueError(f"Undefined phase condition: {self.world_state['action_phase']}")

        return False

    def _compute_reinforcements(self) -> int:

        CLASSIC_MAP_SIZE = 42
        DEFAULT_DENSITY = 2.8

        # Troop reinforcements have been scaled with map size for the sake of balancing.
        if not self.has_received_starting_reinforcement[self.current_agent_idx]:
            self.has_received_starting_reinforcement[self.current_agent_idx] = True
            self.is_first_turn = not np.all(self.has_received_starting_reinforcement)
            if self.n_agents in self._STARTING_REINFORCEMENTS:
                starting_armies = round(
                    self._STARTING_REINFORCEMENTS[self.n_agents]
                    * self.num_nodes
                    / CLASSIC_MAP_SIZE
                )
            else:
                starting_armies = round(
                    DEFAULT_DENSITY
                    * self.num_nodes
                    / self.n_agents
                )
            return starting_armies

        troops_owned_territories = self.territory_counts[self.current_agent_idx]
        territory_armies = max(
            1,
            round(self.num_nodes / 14),
            round(troops_owned_territories * 14 / self.num_nodes)
        )

        continents_armies = 0
        for mask, bonus in self.continent_data:
            if np.all(self.world_state["territory_owner"][mask] == self.current_agent_idx):
                continents_armies += bonus

        total_amount = territory_armies + continents_armies
        return total_amount

    def _update_cards(self, action : int | TradeChoices):
        match action:
            case TradeChoices.NO_OP:
                return
            case TradeChoices.TRADE_ARTILLERY:
                self.world_state["cards_in_hand"][self.current_agent_idx][CardTypes.ARTILLERY] -= 3
            case TradeChoices.TRADE_INFANTRY:
                self.world_state["cards_in_hand"][self.current_agent_idx][CardTypes.INFANTRY] -= 3
            case TradeChoices.TRADE_CAVALRY:
                self.world_state["cards_in_hand"][self.current_agent_idx][CardTypes.CAVALRY] -= 3
            case TradeChoices.TRADE_MIXED:
                self.world_state["cards_in_hand"][self.current_agent_idx][CardTypes.ARTILLERY] -= 1
                self.world_state["cards_in_hand"][self.current_agent_idx][CardTypes.INFANTRY] -= 1
                self.world_state["cards_in_hand"][self.current_agent_idx][CardTypes.CAVALRY] -= 1
            case TradeChoices.TRADE_JOKER:
                has_two_cards = self.world_state["cards_in_hand"][self.current_agent_idx] >= 2
                self.world_state["cards_in_hand"][self.current_agent_idx][CardTypes.JOKER] -= 1
                for card_type in CardTypes:
                    if card_type == CardTypes.JOKER:
                        continue
                    if has_two_cards[card_type]:
                        self.world_state["cards_in_hand"][self.current_agent_idx][card_type] -= 2
                        break
            case _:
                raise ValueError(f"Undefined trade choice: {action}")

        return


    def _update_strength(self):

        territory_coeff = 1.0
        troops_coeff = 1.0
        continent_coeff = 5.0

        territory_owner = self.world_state["territory_owner"]

        territory_strength = territory_coeff * self.territory_counts / self.num_nodes

        troops_strength = troops_coeff * np.log1p(self.troop_counts) / self.log_max_troops

        continent_strength = np.zeros(self.n_agents, dtype=np.float32)

        for mask, bonus in self.continent_data:

            owners = territory_owner[mask]

            counts = np.bincount(
                owners[owners >= 0],
                minlength=self.n_agents
            )

            continent_strength += (
                bonus
                * self._continent_lookup[np.count_nonzero(mask)][counts]
            )

        continent_strength = continent_coeff * continent_strength / self.max_continent_bonus

        max_possible_strength = territory_coeff + troops_coeff + continent_coeff

        strength = (
            territory_strength
            + troops_strength
            + continent_strength
        ) / max_possible_strength

        phi = strength

        delta_phi = self.gamma*phi - self.prev_phi
        raw_reward = delta_phi

        raw_reward *= 0.5

        self.prev_phi[:] = phi

        for i in range(self.n_agents):
            self.rewards[self.agents[i]] += float(raw_reward[i])

        self.strength_dirty = False

    

    def _compute_observation(self, agent : str) -> dict:
        agent_idx = self.agent_to_idx[agent]

        ws = self.world_state

        obs = {
            "territory_owner": ws["territory_owner"],
            "number_of_armies": ws["number_of_armies"],
            "action_phase": np.array(ws["action_phase"], dtype=np.int8),
            "troops_to_place": np.array(ws["troops_to_place"], dtype=np.int16),
            "selected_node": np.array(ws["selected_node"], dtype=np.int16),
            "selected_edge": np.array(ws["selected_edge"], dtype=np.int16),
        }

        owners = obs["territory_owner"]
        obs["territory_owner"] = np.where(
            owners >= 0,
            (owners - agent_idx) % self.n_agents,
            owners
        ).astype(np.int8)

        if self.is_card_game:
            cards = ws["cards_in_hand"]

            obs["cards_in_hand"] = cards[agent_idx].copy()

            obs["amount_cards_others"] = np.sum(cards, axis=1).astype(np.int16)
            obs["amount_cards_others"] = np.delete(obs["amount_cards_others"], agent_idx)

        action_mask = self.risk_helper.generate_action_mask(
            agent_state=ws,
            agent_id=agent_idx
        )

        observation = {
            "observation": obs,
            "action_mask": action_mask
        }

        return observation