# https://github.com/thu-ml/tianshou/blob/master/examples/vizdoom/vizdoom_ppo.py
import numpy as np
import copy
import os
import torch
import tianshou as ts
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

from models.GNN import GraphNetwork

# NOTE: This code has been refactored using AI (specifically the LLM gpt-oss:20b).
# The original code to refactor was:
# - Partially taken by Tianshou 2.0.1's documentation: https://tianshou.org/en/stable/02_deep_dives/L6_MARL.html
# - Created and adapted by me to work on the custom Risk environment and custom model.
# To see the original code, check the git history (specifically the one preceeding this update: 0754e36859e3d3b1eb74a2af7abcdefe9ccd2a8f).

class MARLRandomDiscreteMaskedOnPolicyAdapter(MARLRandomDiscreteMaskedOffPolicyAlgorithm):
    def _update_with_batch(self, batch, batch_size=None, repeat=None, **kwargs):
        return super()._update_with_batch(batch)

class Args:
    # Core hyper‑parameters --------------------------------------------------
    seed = 1626

    lr = 1e-4
    gamma = 0.995
    gae_lambda = 0.95
    vf_coef = 0.4

    start_ent_coef = 3e-2
    end_ent_coef = 1e-2
    ent_coef_decay_period = 0.5

    # Training schedule -------------------------------------------------------
    max_epochs_phase = 10
    epoch_num_steps = 2**17

    collection_step_num_env_steps = 16384
    update_per_step = 1.0

    batch_size = 512
    num_train_envs = 16
    num_test_envs = 16

    max_grad_norm = 0.5
    eps_clip = 0.2

    dual_clip: float | None = None
    value_clip = True
    advantage_normalization = True
    recompute_advantage = True
    return_scaling = True

    # Logging / I/O ----------------------------------------------------------
    logdir = "log/ppo/"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_interval_epochs = 1
    update_step_num_repetitions = 2

    # Derived / convenience --------------------------------------------------
    buffer_size = collection_step_num_env_steps * 4

args = Args()

class PhaseConfig:
    def __init__(self, map_path, max_iters, n_agents, opponents):
        self.map_path = map_path
        self.max_iters = max_iters
        self.n_agents = n_agents
        self.opponents = opponents

def get_env(env_config):
    from environment import risk
    raw_env = risk.env(
        n_agents=env_config["n_agents"],
        max_iters=env_config["max_iters"],
        map_path=env_config["map_path"],
        is_card_game=env_config["is_card_game"],
        max_armies=env_config["max_armies"],
        dense_rewards=env_config["dense_rewards"],
        is_blitz=env_config["is_blitz"],
        gamma=args.gamma
    )
    return PettingZooEnv(raw_env)

def build_agents(env, net_actor, net_critic, opponent_types, phase_index):
    """
    Build a list of agents for the current phase.
    * Agent 0 (the learner) is created once and will load a checkpoint from the previous
      phase if it exists; otherwise it keeps the freshly‑initialised PPO policy.
    * For every opponent type listed in `opponent_types` we either:
        - add a random agent,
        - try to load a checkpoint from the previous phase (if the file exists).
      If no checkpoint is found for `previous_checkpoint`, the code falls back to a
      random agent so that training can still run.
    """
    # Base PPO policy – created only once
    base_policy = DiscreteActorPolicy(actor=net_actor, action_space=env.action_space)
    base_algo = PPO(
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
        return_scaling=args.return_scaling
    ).to(args.device)

    # Load checkpoint for agent 0 if it exists
    ckpt_path = os.path.join(args.logdir, f"ppo_{phase_index-1}", "actor.pth")
    if os.path.exists(ckpt_path):
        base_algo.load_state_dict(torch.load(ckpt_path))

    agents = [base_algo]

    n_other = env.num_agents - 1
    for i in range(n_other):
        typ = opponent_types[i % len(opponent_types)]

        match typ:
            case "random":
                agents.append(MARLRandomDiscreteMaskedOnPolicyAdapter(action_space=env.action_space))

            case "previous_checkpoint":
                actor_copy = copy.deepcopy(net_actor)
                critic_copy = copy.deepcopy(net_critic)

                policy_copy = DiscreteActorPolicy(actor=actor_copy, action_space=env.action_space)
                algo_copy = PPO(
                    policy=policy_copy,
                    critic=critic_copy,
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
                    return_scaling=args.return_scaling
                ).to(args.device)

                if os.path.exists(ckpt_path):
                    algo_copy.load_state_dict(torch.load(ckpt_path))
                else:
                    # No checkpoint – fall back to a random agent
                    algo_copy = MARLRandomDiscreteMaskedOnPolicyAdapter(action_space=env.action_space)

                agents.append(algo_copy)

            case _:
                # Unknown type – treat as random
                agents.append(MARLRandomDiscreteMaskedOnPolicyAdapter(action_space=env.action_space))

    return base_algo, agents


def create_collectors(marl_algorithm, train_envs, test_envs):
    training_collector = Collector(
        marl_algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=False
    )
    test_collector = Collector(marl_algorithm, test_envs, exploration_noise=False)
    return training_collector, test_collector

def train_phase(phase_index, phase_cfg, writer):
    env_params = {
        "max_iters": phase_cfg.max_iters,
        "n_agents": phase_cfg.n_agents,
        "map_path": phase_cfg.map_path,
        "is_card_game": True,
        "max_armies": 5000,
        "dense_rewards": True,
        "is_blitz" : True,
        "gamma" : args.gamma
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

    net_actor = GraphNetwork(
        obs_space=env.observation_space,
        action_space=env.action_space,
        map_graph=env.env.map_network,
        res_hidden_size=128,
        gnn_hidden_size=32,
        graph_depth=2,
        residual_depth=2,
        embed_space_phase=8,
        activation_function=torch.nn.SiLU,
        starting_residual_scale=0.2,
        n_heads=4
    ).to(args.device)

    net_critic = GraphNetwork(
        obs_space=env.observation_space,
        action_space=1,
        map_graph=env.env.map_network,
        res_hidden_size=128,
        gnn_hidden_size=32,
        graph_depth=1,
        residual_depth=2,
        embed_space_phase=8,
        activation_function=torch.nn.SiLU,
        starting_residual_scale=0.2,
        n_heads=4
    ).to(args.device)

    algorithm, agents = build_agents(env, net_actor, net_critic, phase_cfg.opponents, phase_index)
    marl_algorithm = MultiAgentOnPolicyAlgorithm(algorithms=agents, env=env)

    training_collector, test_collector = create_collectors(marl_algorithm, train_envs, test_envs)

    target_agent_name = env.agents[0]
    target_agent_index = env.agents.index(target_agent_name)

    def reward_metric(rews: np.ndarray) -> np.ndarray:
        return rews[:, target_agent_index].flatten()

    def save_best_fn(policy):
        model_save_path = os.path.join(args.logdir, f"ppo_{phase_index}")
        os.makedirs(model_save_path, exist_ok=True)
        alg = policy.get_algorithm(target_agent_name)
        torch.save(alg.state_dict(), os.path.join(model_save_path, "actor.pth"))
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
            1.0, epoch / (args.max_epochs_phase * args.ent_coef_decay_period))
        algorithm.ent_coef = new_ent

    result = marl_algorithm.run_training(
        OnPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=args.max_epochs_phase,
            epoch_num_steps=args.epoch_num_steps,
            update_step_num_repetitions=args.update_step_num_repetitions,
            test_step_num_episodes=args.num_test_envs,
            batch_size=args.batch_size,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            training_fn=train_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_training=False,
            multi_agent_return_reduction=reward_metric
        )
    )
    print(f"Phase {phase_index} result: ", result)

def main():
    phases = [
        PhaseConfig(
            map_path=f"./environment/maps/simplified.json",
            max_iters=3000,
            n_agents=2,
            opponents=["random"]
        ),
        PhaseConfig(
            map_path=f"./environment/maps/simplified.json",
            max_iters=3000,
            n_agents=4,
            opponents=["random", "previous_checkpoint"]
        ),
        PhaseConfig(
            map_path=f"./environment/maps/classic.json",
            max_iters=5000,
            n_agents=3,
            opponents=["random"]
        ),
        PhaseConfig(
            map_path=f"./environment/maps/classic.json",
            max_iters=5000,
            n_agents=4,
            opponents=["random", "previous_checkpoint"]
        )
    ]

    writer = SummaryWriter(os.path.join(args.logdir, "ppo_all"))
    for idx, phase_cfg in enumerate(phases):
        train_phase(idx, phase_cfg, writer)
    writer.close()

if __name__ == "__main__":
    main()