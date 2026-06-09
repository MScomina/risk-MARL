# https://gymnasium.farama.org/api/env/
# https://pettingzoo.farama.org/content/environment_creation/

from copy import copy
import functools

import numpy as np

from gymnasium.utils import seeding, EzPickle
from gymnasium.spaces import Dict
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from pathlib import Path

from .maps import graph_utils
from .risk_utils import CardTypes, FlattenObservationWrapper, RiskPhase, RiskHelper, TradeChoices

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
MAX_ARMIES = 100
MAX_ITERS = 10_000
IS_CARD_GAME = True
DENSE_REWARDS = True

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
                 dense_rewards : bool = DENSE_REWARDS):

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

        self.risk_helper = RiskHelper(
            game_map=self.map_network,
            num_agents=self.n_agents,
            max_armies=self.max_armies,
            is_card_game=self.is_card_game
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

        self.continent_masks = self.risk_helper._generate_continent_masks()
        self.max_continent_bonus = sum(
            c["bonus"] for c in self.map_network.graph["continents"].values()
        )
        self.continent_data = [
            (self.continent_masks[name], data["bonus"])
            for name, data in self.map_network.graph["continents"].items()
        ]

        self._continent_lookup = {}

        for mask, _ in self.continent_data:
            n = np.count_nonzero(mask)

            if n not in self._continent_lookup:
                p = np.arange(n + 1) / n
                self._continent_lookup[n] = (
                    np.exp(6.0 * p) - 1
                ) / (
                    np.exp(6.0) - 1
                )

        self._continent_strength_buffer = np.zeros(
            self.n_agents,
            dtype=np.float32
        )

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.current_agent_idx = 0

        self.render_mode = render_mode


    @functools.lru_cache(maxsize=1000000)
    def observation_space(self, agent):
        return self._observation_spaces[agent]


    @functools.lru_cache(maxsize=1000000)
    def action_space(self, agent):
        return self._action_spaces[agent]


    def reset(self, seed=None, options=None):

        self.timestep = 0

        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        self.agents = copy(self.possible_agents)
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.world_state = self.risk_helper.starting_observation(full_knowledge=True)

        self.state = {
            agent : self.world_state for agent in self.agents
        }

        self.observations = {
            agent : {
                "observation": self.world_state,
                "action_mask": self.risk_helper.generate_action_mask(
                    agent_state=self.world_state,
                    agent_id=idx
                )
            } for idx, agent in enumerate(self.agents)
        }

        self.num_moves = 0

        self.has_conquered_this_turn = False
        self.is_first_turn = True

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.agent_to_idx = {
            agent: i
            for i, agent in enumerate(self.agents)
        }
        self.current_agent_idx = self.agent_to_idx[self.agent_selection]
        self.strength = np.zeros(self.n_agents)
        self.territory_counts = np.zeros(self.n_agents, dtype=np.int16)
        self.troop_counts = np.zeros(self.n_agents, dtype=np.int32)


    def observe(self, agent: str):
        agent_idx = self.agent_to_idx[agent]

        ws = self.world_state

        obs = {
            "territory_owner": ws["territory_owner"].copy(),
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

        self.observations[agent] = {
            "observation": obs,
            "action_mask": action_mask
        }

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
        if self.dense_rewards:
            self._update_strength()

        if self.territory_counts[self.current_agent_idx] == self.num_nodes:
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
                self.world_state["troops_to_place"] = self._compute_reinforcements(self.agent_selection)
            self.has_conquered_this_turn = False

        self.num_moves += 1
        if self.num_moves >= self.max_iters:
            for agent in self.agents:
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
                self.world_state["action_phase"] = RiskPhase.SELECT_ARMY_COUNT
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
                edge = self.map_network.graph["idx_to_edge"][action]
                src_node = self.map_network.graph["node_to_idx"][edge[0]]
                if self.world_state["territory_owner"][src_node] != self.current_agent_idx:
                    raise RuntimeError(f"Agent {agent} tried to pick an invalid edge ({src_node} is owned by {self.world_state['territory_owner'][src_node]}): {action}")
                self.world_state["selected_edge"] = action
                self.world_state["action_phase"] = RiskPhase.SELECT_ARMY_COUNT
                return False

            case RiskPhase.SELECT_ARMY_COUNT:

                if self.world_state["selected_node"] != -1:
                    # Reinforcing own territory
                    if self.world_state["troops_to_place"] < action:
                        raise RuntimeError(f"Agent {agent} tried to reinforce with more troops than available: {action}")
                    self.world_state["number_of_armies"][self.world_state["selected_node"]] += action
                    self.troop_counts[self.current_agent_idx] += action
                    self.world_state["troops_to_place"] -= action
                    if self.world_state["troops_to_place"] > 0 or self.is_first_turn:
                        self.world_state["action_phase"] = RiskPhase.SELECT_NODE
                        if self.world_state["troops_to_place"] == 0:
                            if self._agent_selector.is_last():
                                # Done all the players' first turns, time to start the actual normal reinforcements
                                self.is_first_turn = False
                            # Still in starting reinforcement, switching to another player.
                            self.world_state["selected_node"] = -1
                            return True
                    else:
                        self.world_state["action_phase"] = RiskPhase.SELECT_EDGE
                    self.world_state["selected_node"] = -1
                    return False

                elif self.world_state["selected_edge"] != -1:

                    edge = self.map_network.graph["idx_to_edge"][self.world_state["selected_edge"]]
                    src_node = self.map_network.graph["node_to_idx"][self.map_network.graph["idx_to_edge"][self.world_state["selected_edge"]][0]]
                    dst_node = self.map_network.graph["node_to_idx"][self.map_network.graph["idx_to_edge"][self.world_state["selected_edge"]][1]]

                    if self.world_state["territory_owner"][dst_node] != self.current_agent_idx:
                        # Attacking.
                        if action == 0 or action > min(3,self.world_state["number_of_armies"][src_node]-1):
                            raise RuntimeError(f"Agent {agent} tried to attack with too many troops: {action}")

                        previous_owner = self.world_state["territory_owner"][dst_node]

                        dst_amount = self.world_state["number_of_armies"][dst_node]
                        atk_losses, def_losses = self.risk_helper.risk_attack_outcome(action, dst_amount)

                        self.world_state["number_of_armies"][src_node] -= atk_losses
                        self.world_state["number_of_armies"][dst_node] -= def_losses

                        self.troop_counts[self.current_agent_idx] -= atk_losses
                        self.troop_counts[previous_owner] -= def_losses

                        if self.world_state["number_of_armies"][dst_node] == 0:
                            # No troops left, control switches.
                            if not self.has_conquered_this_turn and self.is_card_game:
                                self.world_state["cards_in_hand"][self.current_agent_idx][self.risk_helper.draw_card()] += 1
                                self.has_conquered_this_turn = True
                            self.territory_counts[self.current_agent_idx] += 1
                            self.territory_counts[previous_owner] -= 1
                            self.world_state["territory_owner"][dst_node] = self.current_agent_idx
                            self.world_state["number_of_armies"][dst_node] += action - atk_losses
                            self.world_state["number_of_armies"][src_node] -= action - atk_losses
                            if self.territory_counts[previous_owner] <= 0:
                                self.terminations[self.agents[previous_owner]] = True
                                if self.is_card_game:
                                    self.world_state["cards_in_hand"][self.current_agent_idx] += self.world_state["cards_in_hand"][previous_owner]
                                    self.world_state["cards_in_hand"][previous_owner] = np.zeros(len(CardTypes), dtype=np.int16)
                                if self.dense_rewards:
                                    self.rewards[agent] += 0.1 / (self.num_agents-1)    # Wiping opponent is good
                                    self.rewards[self.agents[previous_owner]] -= 1.0 # Dying is bad

                        self.world_state["action_phase"] = RiskPhase.SELECT_EDGE
                        self.world_state["selected_edge"] = -1
                        return False

                    else:
                        # Moving.
                        if action == 0 or action >= self.world_state["number_of_armies"][src_node]:
                            raise RuntimeError(f"Agent {agent} tried to move with too many troops: {action}")
                        self.world_state["number_of_armies"][src_node] -= action
                        self.world_state["number_of_armies"][dst_node] += action
                        if self.is_card_game:
                            self.world_state["action_phase"] = RiskPhase.TRADE_CARDS
                        else:
                            self.world_state["action_phase"] = RiskPhase.SELECT_NODE
                        self.world_state["selected_edge"] = -1
                        return True

            case RiskPhase.TRADE_CARDS:
                self.world_state["troops_to_place"] += self.risk_helper.cards_trade_amount(action)
                self._update_cards(action=action)
                self.world_state["action_phase"] = RiskPhase.SELECT_NODE
                return False

            case _:
                raise ValueError(f"Undefined phase condition: {self.world_state['action_phase']}")

        return False

    def _compute_reinforcements(self, agent : str) -> int:

        if self.is_first_turn:
            if self.n_agents in _STARTING_REINFORCEMENTS:
                return _STARTING_REINFORCEMENTS[self.n_agents]
            return 120//self.n_agents

        troops_owned_territories = self.territory_counts[self.current_agent_idx]
        territory_armies = max(3, troops_owned_territories//3)

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

        if self.world_state["action_phase"] == RiskPhase.STARTING_PLACEMENT or self.is_first_turn:
            return np.zeros(self.n_agents, dtype=np.float32)

        territory_coeff = 1.0
        troops_coeff = 0.5
        continent_coeff = 1.0

        territory_owner = self.world_state["territory_owner"]
        armies = self.world_state["number_of_armies"]

        territory_strength = territory_coeff * self.territory_counts / self.num_nodes

        troops_strength = troops_coeff * np.log1p(self.troop_counts) / self.log_max_troops

        continent_strength = np.zeros(self.n_agents, dtype=np.float32)

        continent_spiky = 6.0

        continent_exp_denom = np.exp(continent_spiky) - 1

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

        strength = territory_strength + troops_strength + continent_strength

        total_strength = np.sum(strength)

        relative_strength = (
            strength
            - (total_strength - strength) / (self.n_agents - 1)
        )

        max_possible_strength = territory_coeff + troops_coeff + continent_coeff

        reward_scale_strength = 0.5
        rewards = reward_scale_strength * (
            relative_strength - self.strength
        ) / max_possible_strength

        agents = self.agents

        for i in range(self.n_agents):
            self.rewards[agents[i]] += float(rewards[i])

        self.strength[:] = relative_strength