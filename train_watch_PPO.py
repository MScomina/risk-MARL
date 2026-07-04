# https://github.com/thu-ml/tianshou/blob/master/examples/vizdoom/vizdoom_ppo.py
# https://arxiv.org/pdf/2006.05990 (What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study)
# https://link.springer.com/content/pdf/10.1007/s10994-025-06822-0.pdf (Analyzing the effect of residual connections to oversmoothing in graph neural networks)

import numpy as np
import copy
import os
import random
import torch
import tianshou as ts
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.discrete import DiscreteActor
from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.algorithm.random import MARLRandomDiscreteMaskedOffPolicyAlgorithm
from tianshou.algorithm import PPO
from tianshou.env import PettingZooEnv
from tianshou.algorithm.multiagent.marl import MultiAgentOnPolicyAlgorithm
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OnPolicyTrainerParams

from agents.models.GNN import GraphNetwork

# NOTE: This code has been refactored using AI (specifically the LLM gpt-oss:20b).
# The original code to refactor was:
# - Partially taken by Tianshou 2.0.1's documentation: https://tianshou.org/en/stable/02_deep_dives/L6_MARL.html
# - Created and adapted by me to work on the custom Risk environment and custom model.
# To see the original code, check the git history (specifically the one preceeding this update: 0754e36859e3d3b1eb74a2af7abcdefe9ccd2a8f).

class MARLRandomDiscreteMaskedOnPolicyAdapter(MARLRandomDiscreteMaskedOffPolicyAlgorithm):
    """Convenience wrapper that behaves like an on‑policy algorithm."""
    def _update_with_batch(self, batch, batch_size=None, repeat=None, **kwargs):
        return super()._update_with_batch(batch)

class Args:
    # Core hyper‑parameters --------------------------------------------------
    seed = 1626

    lr = 2e-4
    gamma = 0.995
    gae_lambda = 0.97
    vf_coef = 0.5

    start_ent_coef = 1e-2
    end_ent_coef = 1e-3
    ent_coef_decay_period = 0.66
    update_step_num_repetitions = 5

    # Training schedule -------------------------------------------------------
    epoch_num_steps = 2**17

    collection_step_num_env_steps = 4096
    update_per_step = 1.0

    batch_size = 256
    num_train_envs = 16
    num_test_envs = 16

    max_grad_norm = 0.5
    eps_clip = 0.2

    dual_clip: float | None = None
    value_clip = True
    advantage_normalization = True
    recompute_advantage = True
    return_scaling = True

    starting_phase = 0

    # Logging / I/O ----------------------------------------------------------
    logdir = "log/ppo/"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_interval_epochs = 1

    # Derived / convenience --------------------------------------------------
    buffer_size = collection_step_num_env_steps * 4

    # What to do ---------------------------------------------
    train = False   # run training (default)
    watch = True    # render 5 episodes with the last checkpoint

args = Args()

# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PhaseConfig:
    map_path : str
    max_iters : int
    n_agents : int
    type_opponents : list[str]
    n_epochs : int

def get_env(env_config):
    """Create a PettingZoo wrapper around the custom Risk environment."""
    from environment import risk

    raw_env = risk.env(
        n_agents=env_config["n_agents"],
        max_iters=env_config["max_iters"],
        map_path=env_config["map_path"],
        is_card_game=env_config["is_card_game"],
        max_armies=env_config["max_armies"],
        dense_rewards=env_config["dense_rewards"],
        is_blitz=env_config["is_blitz"],
        render_mode=env_config.get("render_mode", None),
        gamma=args.gamma,
    )
    return PettingZooEnv(raw_env)

# --------------------------------------------------------------------------- #
def build_agents(env, net_actor, net_critic, opponent_types, phase_index):
    """
    Build a list of agents for the current phase.
    * Agent 0 (the learner) is created once and will load a checkpoint from the
      previous phase if it exists; otherwise it keeps the freshly‑initialised PPO policy.
    * For every opponent type listed in `opponent_types` we either:
        - add a random agent,
        - try to load a checkpoint from the previous phase (if the file exists).
      If no checkpoint is found for `previous_checkpoint`, the code falls back
      to a random agent so that training can still run.
    """
    # Base PPO policy – created only once
    ckpt_actor_path = os.path.join(args.logdir, f"ppo_{phase_index-1}", "actor.pth")
    if os.path.exists(ckpt_actor_path):
        net_actor.load_state_dict(torch.load(ckpt_actor_path))
    ckpt_critic_path = os.path.join(args.logdir, f"ppo_{phase_index-1}", "critic.pth")
    if os.path.exists(ckpt_critic_path):
        net_critic.load_state_dict(torch.load(ckpt_critic_path))
    base_policy = DiscreteActorPolicy(actor=net_actor, action_space=env.action_space)
    base_algo = ts.algorithm.PPO(
        policy=base_policy,
        critic=net_critic,
        optim=AdamOptimizerFactory(lr=args.lr),
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.start_ent_coef,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        advantage_normalization=args.advantage_normalization,
        recompute_advantage=args.recompute_advantage,
        eps_clip=args.eps_clip,
        return_scaling=args.return_scaling,
    )

    agents = [base_algo]

    n_other = env.num_agents - 1
    for i in range(n_other):
        typ = random.choice(opponent_types)

        match typ:
            case "random":
                agents.append(
                    MARLRandomDiscreteMaskedOnPolicyAdapter(action_space=env.action_space)
                )

            case "previous_checkpoint":
                # Frozen previous checkpoint.
                critic_copy = copy.deepcopy(net_critic)
                critic_copy.eval()

                actor_copy = copy.deepcopy(net_actor)
                actor_copy.eval()

                if random.random() < 0.5:
                    # Mix in some random previous checkpoints.
                    previous_actor_path = os.path.join(args.logdir, f"ppo_{phase_index//2}", "actor.pth")
                    previous_critic_path = os.path.join(args.logdir, f"ppo_{phase_index//2}", "critic.pth")
                    if os.path.exists(previous_actor_path):
                        actor_copy.load_state_dict(torch.load(previous_actor_path))
                    if os.path.exists(previous_critic_path):
                        critic_copy.load_state_dict(torch.load(previous_critic_path))

                policy_copy = DiscreteActorPolicy(actor=actor_copy, action_space=env.action_space)
                policy_copy.eval()
                
                algo_copy = ts.algorithm.PPO(
                    policy=policy_copy,
                    critic=critic_copy,
                    optim=AdamOptimizerFactory(lr=0.0),
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    max_grad_norm=args.max_grad_norm,
                    vf_coef=args.vf_coef,
                    ent_coef=args.start_ent_coef,
                    dual_clip=args.dual_clip,
                    value_clip=args.value_clip,
                    advantage_normalization=args.advantage_normalization,
                    recompute_advantage=args.recompute_advantage,
                    eps_clip=args.eps_clip,
                    return_scaling=args.return_scaling,
                )
                algo_copy.eval()

                agents.append(algo_copy)

            case _:
                # Unknown type – treat as random
                agents.append(
                    MARLRandomDiscreteMaskedOnPolicyAdapter(action_space=env.action_space)
                )

    return base_algo, agents

# --------------------------------------------------------------------------- #
def create_collectors(marl_algorithm, train_envs, test_envs):
    training_collector = Collector(
        policy=marl_algorithm,
        env=train_envs,
        buffer=None,
        exploration_noise=False,
    )
    test_collector = Collector(marl_algorithm, test_envs, exploration_noise=False)
    return training_collector, test_collector

# --------------------------------------------------------------------------- #
def init_actor_critic(petting_zoo_env):
    """
    Create the actor and critic GraphNetwork instances for a given PettingZooEnv.
    The same configuration is used in both training and watching.
    """
    net_actor = GraphNetwork(
        obs_space=petting_zoo_env.observation_space,
        action_space=petting_zoo_env.action_space,
        map_graph=petting_zoo_env.env.map_network,
        res_hidden_size=128,
        gnn_hidden_size=48,
        graph_depth=3,
        residual_depth=4,
        policy_head_depth=5,
        embed_space_phase=8,
        activation_function=torch.nn.SiLU,
        starting_residual_scale=0.6
    ).to(args.device)

    net_critic = GraphNetwork(
        obs_space=petting_zoo_env.observation_space,
        action_space=1,  # critic outputs a scalar
        map_graph=petting_zoo_env.env.map_network,
        res_hidden_size=128,
        gnn_hidden_size=48,
        graph_depth=3,
        residual_depth=4,
        embed_space_phase=8,
        activation_function=torch.nn.SiLU,
        starting_residual_scale=0.6
    ).to(args.device)

    return net_actor, net_critic

# --------------------------------------------------------------------------- #
def train_phase(phase_index, phase_cfg, writer):
    env_params = {
        "max_iters": phase_cfg.max_iters,
        "n_agents": phase_cfg.n_agents,
        "map_path": phase_cfg.map_path,
        "is_card_game": True,
        "max_armies": 5000,
        "dense_rewards": True,
        "is_blitz": True,
        "gamma": args.gamma,
    }

    train_envs = ts.env.SubprocVectorEnv(
        [(lambda cfg=env_params: get_env(cfg)) for _ in range(args.num_train_envs)]
    )
    test_envs = ts.env.SubprocVectorEnv(
        [(lambda cfg=env_params: get_env(cfg)) for _ in range(args.num_test_envs)]
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    env = get_env(env_params)

    logger = TensorboardLogger(writer)

    net_actor, net_critic = init_actor_critic(env)

    algorithm, agents = build_agents(env, net_actor, net_critic, phase_cfg.type_opponents, phase_index)
    marl_algorithm = MultiAgentOnPolicyAlgorithm(algorithms=agents, env=env)

    training_collector, test_collector = create_collectors(marl_algorithm, train_envs, test_envs)

    target_agent_idx = 0

    def reward_metric(rews: np.ndarray) -> np.ndarray:
        return rews[:, target_agent_idx].flatten()

    def save_best_fn(policy):
        model_save_path = os.path.join(args.logdir, f"ppo_{phase_index}")
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(net_actor.state_dict(), os.path.join(model_save_path, "actor.pth"))
        torch.save(net_critic.state_dict(), os.path.join(model_save_path, "critic.pth"))

    def train_fn(epoch: int, env_step: int):
        if (epoch % args.log_interval_epochs == 0) and (env_step % args.epoch_num_steps == 0):
            for name, param in net_actor.named_parameters():
                writer.add_histogram(f"params_act/{name}", param.clone().detach().cpu(), global_step=env_step)
                if param.grad is not None:
                    writer.add_histogram(f"grads_act/{name}", param.grad.clone().detach().cpu(), global_step=env_step)

            for name, param in net_critic.named_parameters():
                writer.add_histogram(f"params_crit/{name}", param.clone().detach().cpu(), global_step=env_step)
                if param.grad is not None:
                    writer.add_histogram(f"grads_crit/{name}", param.grad.clone().detach().cpu(), global_step=env_step)

        new_ent = args.start_ent_coef - (args.start_ent_coef - args.end_ent_coef) * min(
            1.0,
            epoch / (phase_cfg.n_epochs * args.ent_coef_decay_period),
        )
        algorithm.ent_coef = new_ent

    result = marl_algorithm.run_training(
        OnPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=phase_cfg.n_epochs,
            epoch_num_steps=args.epoch_num_steps,
            update_step_num_repetitions=args.update_step_num_repetitions,
            test_step_num_episodes=args.num_test_envs,
            batch_size=args.batch_size,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            training_fn=train_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_training=False,
            multi_agent_return_reduction=reward_metric,
        )
    )
    print(f"Phase {phase_index} result: ", result)

# --------------------------------------------------------------------------- #
def watch(last_phase_cfg, phase_index):
    """
    Render a single episode with the last trained policy.

    * `last_phase_cfg` – configuration of the final training phase.
    * `phase_index`   – index that will be passed to `build_agents`; it must be
      one larger than the numeric part used when saving checkpoints (so
      `phase_index = len(phases)`).
    """
    env_params = {
        "max_iters": last_phase_cfg.max_iters,
        "n_agents": last_phase_cfg.n_agents,
        "map_path": last_phase_cfg.map_path,
        "is_card_game": True,
        "max_armies": 10000,
        "dense_rewards": True,
        "is_blitz": True,
        "gamma": args.gamma,
        "render_mode": "human",
    }

    raw_env = get_env(env_params)
    env = ts.env.SubprocVectorEnv([(lambda cfg=env_params: get_env(cfg))])

    net_actor, net_critic = init_actor_critic(raw_env)

    net_actor.eval()
    net_critic.eval()

    # Build agents using the same logic as training – this will load checkpoints
    # from `ppo_{phase_index-1}` for every agent that has a "previous_checkpoint"
    # entry in the opponent list.
    _, agents = build_agents(raw_env, net_actor, net_critic, last_phase_cfg.type_opponents, phase_index)

    marl_algorithm = MultiAgentOnPolicyAlgorithm(algorithms=agents, env=raw_env)

    collector = Collector(marl_algorithm, env)
    # Render episodes
    with torch.no_grad():
        result = collector.collect(
            n_episode=5,
            reset_before_collect=True,
            render=0.1
        )

# --------------------------------------------------------------------------- #
def main():
    phases = [
        PhaseConfig(
            map_path="./environment/maps/classic.json",
            max_iters=2000,
            n_agents=2,
            type_opponents=["random"],
            n_epochs=10
        ),        
        PhaseConfig(
            map_path="./environment/maps/classic.json",
            max_iters=3000,
            n_agents=4,
            type_opponents=["random"],
            n_epochs=20
        ),
        *[PhaseConfig(
            map_path="./environment/maps/classic.json",
            max_iters=3000,
            n_agents=4,
            type_opponents=["random", "previous_checkpoint"],
            n_epochs=10
        ) for _ in range(5)],
        *[PhaseConfig(
            map_path="./environment/maps/classic.json",
            max_iters=3000,
            n_agents=4,
            type_opponents=["previous_checkpoint"],
            n_epochs=10
        ) for _ in range(10)]
    ]

    writer = SummaryWriter(os.path.join(args.logdir, "ppo_all"))

    if args.train:
        for idx, phase_cfg in enumerate(phases):
            if idx < args.starting_phase:
                continue
            train_phase(idx, phase_cfg, writer)
        writer.close()

    # After training (or even independently) we can watch the last policy
    if args.watch:
        # `phase_index` must be one larger than the numeric part used when saving checkpoints
        watch(phases[-1], len(phases))

if __name__ == "__main__":
    main()