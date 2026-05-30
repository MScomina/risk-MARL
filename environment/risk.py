# https://gymnasium.farama.org/api/env/
# https://pettingzoo.farama.org/content/environment_creation/

from copy import copy, deepcopy
import functools

import numpy as np

from gymnasium.utils import seeding
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
#       - Action phase (0: initial placing, 1: select army count, 2: select node, 3: select edge)
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

def env(should_flatten_obs : bool = False, **kwargs) -> AECEnv:
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    if should_flatten_obs:
        env = FlattenObservationWrapper(env)
    return env


class raw_env(AECEnv):

    metadata : dict = {"render_modes": ["human"], "name": "risk-v1"}

    def __init__(self, render_mode=None, n_agents : int = NUM_AGENTS, map_path : Path | None = None, 
                 max_armies : int = MAX_ARMIES, max_iters : int = MAX_ITERS, is_card_game : bool = IS_CARD_GAME):

        self.agents = ["player_" + str(r) for r in range(n_agents)]
        self.possible_agents = self.agents[:]
        self.n_agents = n_agents
        self.map_network = graph_utils.generate_graph(map_path)
        self.max_armies = max_armies
        self.max_iters = max_iters
        self.is_card_game = is_card_game

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

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.current_agent_idx = self.agents.index(self.agent_selection)


    def observe(self, agent : str):
        agent_idx = self.agents.index(agent)

        self.observations[agent]["observation"] = deepcopy(self.world_state)
        obs_dict = self.observations[agent]["observation"]

        absolute_owners = obs_dict["territory_owner"]
        relative_owners = np.where(
            absolute_owners >= 0,
            (absolute_owners - agent_idx) % self.num_agents,
            absolute_owners
        )
        obs_dict["territory_owner"] = relative_owners.astype(np.int8)

        obs_dict["action_phase"] = np.array(obs_dict["action_phase"], dtype=np.int8)
        obs_dict["troops_to_place"] = np.array(obs_dict["troops_to_place"], dtype=np.int16)
        obs_dict["selected_node"] = np.array(obs_dict["selected_node"], dtype=np.int16)
        self.observations[agent]["observation"]["selected_edge"] = np.array(obs_dict["selected_edge"], dtype=np.int16)
        if self.is_card_game:
            obs_dict["cards_in_hand"] = self.world_state["cards_in_hand"][agent_idx]
            hand_counts = np.sum(self.world_state["cards_in_hand"], axis=1).astype(np.int16)
            hand_counts = np.delete(hand_counts, agent_idx)
            obs_dict["amount_cards_others"] = hand_counts

        self.observations[agent]["action_mask"] = self.risk_helper.generate_action_mask(
            agent_state=self.world_state,
            agent_id=agent_idx
        )
        return self.observations[agent]


    def close(self):
        pass


    def step(self, action):
        
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        current_agent = self.agent_selection
        self.current_agent_idx = self.agents.index(current_agent)

        current_obs = self.observe(current_agent)
        legal_mask = current_obs["action_mask"]

        self.rewards = {agent: 0 for agent in self.agents}

        should_end_turn = self._update_state(current_agent, action)

        if np.all(self.world_state["territory_owner"] == self.current_agent_idx):
            for agent in self.agents:
                if agent == current_agent:
                    self.rewards[agent] = 1
                else:
                    self.rewards[agent] = -1
                self.terminations[agent] = True
        elif should_end_turn:
            next_agent = self._agent_selector.next()
            while (self.terminations[next_agent] or self.truncations[next_agent]) and next_agent != current_agent:
                if all(self.terminations[a] or self.truncations[a] for a in self.agents):
                    break
                next_agent = self._agent_selector.next()
                
            self.agent_selection = next_agent
            self.current_agent_idx = self.agents.index(self.agent_selection)
            if not self.world_state["action_phase"] == RiskPhase.STARTING_PLACEMENT:
                self.world_state["troops_to_place"] = self._compute_reinforcements(self.agent_selection)
            self.has_conquered_this_turn = False
        
        self.num_moves += 1
        if self._agent_selector.is_last():
            for agent in self.agents:
                self.truncations[agent] = self.num_moves >= self.max_iters

        self._accumulate_rewards()
        self._clear_rewards()

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos


    def _update_state(self, agent : str, action : dict) -> bool:

        # Updates the state and returns whether the turn is over or not.
        match self.world_state["action_phase"]:

            case RiskPhase.STARTING_PLACEMENT:
                if self.world_state["territory_owner"][action] != -1:
                    raise RuntimeError(f"Agent {agent} tried to own an already owned territory: {action}")
                self.world_state["territory_owner"][action] = self.current_agent_idx
                self.world_state["number_of_armies"][action] = 1
                if np.all(self.world_state["territory_owner"] != -1):
                    # All nodes have been taken, reinforcement has begun.
                    self.world_state["action_phase"] = RiskPhase.SELECT_NODE
                return True

            case RiskPhase.SELECT_NODE:
                if action == self.map_network.number_of_nodes():
                    # No-op operation, either because agent is dead or all nodes are filled. Skip to next phase.
                    self.world_state["troops_to_place"] = 0
                    self.world_state["action_phase"] = RiskPhase.SELECT_EDGE
                    return False
                if action > self.map_network.number_of_nodes():
                    raise RuntimeError(f"Agent {agent} tried to pick a node index greater than the number of nodes: {action}")
                if self.world_state["territory_owner"][action] != self.current_agent_idx:
                    raise RuntimeError(f"Agent {agent} tried to pick another player's territory for action: {action}")
                self.world_state["selected_node"] = action
                self.world_state["action_phase"] = RiskPhase.SELECT_ARMY_COUNT
                return False
            
            case RiskPhase.SELECT_EDGE:
                if action == self.map_network.number_of_edges():
                    # No-op operation (no actions available, must pass turn)
                    if self.is_card_game:
                        self.world_state["action_phase"] = RiskPhase.TRADE_CARDS
                    else:
                        self.world_state["action_phase"] = RiskPhase.SELECT_NODE
                    self.world_state["selected_edge"] = -1
                    return True
                if action > self.map_network.number_of_edges():
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
                    self.world_state["troops_to_place"] -= action
                    if self.world_state["troops_to_place"] > 0:
                        self.world_state["action_phase"] = RiskPhase.SELECT_NODE
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

                        dst_amount = self.world_state["number_of_armies"][dst_node]
                        atk_losses, def_losses = self.risk_helper.risk_attack_outcome(action, dst_amount)

                        self.world_state["number_of_armies"][src_node] -= atk_losses
                        self.world_state["number_of_armies"][dst_node] -= def_losses

                        if self.world_state["number_of_armies"][dst_node] == 0:
                            # No troops left, control switches.
                            previous_owner = self.world_state["territory_owner"][dst_node]
                            if not self.has_conquered_this_turn and self.is_card_game:
                                self.world_state["cards_in_hand"][self.current_agent_idx][self.risk_helper.draw_card()] += 1
                                self.has_conquered_this_turn = True
                            self.world_state["territory_owner"][dst_node] = self.current_agent_idx
                            self.world_state["number_of_armies"][dst_node] += action - atk_losses
                            self.world_state["number_of_armies"][src_node] -= action - atk_losses
                            should_terminate = True
                            for owner in self.world_state["territory_owner"]:
                                if owner == previous_owner:
                                    should_terminate = False
                            self.terminations[self.agents[previous_owner]] = should_terminate
                            if should_terminate and self.is_card_game:
                                self.world_state["cards_in_hand"][self.current_agent_idx] += self.world_state["cards_in_hand"][previous_owner]
                                self.world_state["cards_in_hand"][previous_owner] = np.zeros(len(CardTypes), dtype=np.int16)
                        
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

        troops_owned_territories = np.count_nonzero(self.world_state["territory_owner"] == self.current_agent_idx)
        territory_armies = max(3, troops_owned_territories//3)

        continents_armies = 0
        for continent, continent_values in self.map_network.graph["continents"].items():
            amount_continent = continent_values["bonus"]
            continent_mask = self.continent_masks[continent]
            if np.all(self.world_state["territory_owner"][continent_mask] == self.current_agent_idx):
                continents_armies += amount_continent
    
        return (territory_armies + continents_armies)

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