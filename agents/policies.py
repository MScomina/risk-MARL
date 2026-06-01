# https://tianshou.org/en/stable/index.html
# https://tianshou.org/en/stable/02_deep_dives/L6_MARL.html

import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from tianshou.algorithm.multiagent.marl import MultiAgentOffPolicyAlgorithm
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.algorithm.random import MARLRandomDiscreteMaskedOffPolicyAlgorithm
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import PettingZooEnv
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo

from models import ResidualNetwork

class Args:
    seed = 1626
    eps_test = 0.05
    eps_train = 0.1
    buffer_size = 20000
    lr = 1e-4
    gamma = 0.9  # A smaller gamma favors earlier wins
    n_step = 3
    target_update_freq = 320
    epoch = 50
    epoch_num_steps = 1000
    collection_step_num_env_steps = 10
    update_per_step = 0.1
    batch_size = 64
    hidden_sizes = [128, 128, 128, 128]  # noqa: RUF012
    num_train_envs = 10
    num_test_envs = 10
    logdir = "log"
    render = 0.1
    win_rate = 0.6  # Target winning rate (optimal policy can get ~0.7)
    watch = False  # Set to True to skip training and watch pre-trained models
    agent_id = 2  # The learned agent plays as player 2
    resume_path = ""  # Path to pre-trained agent .pth file
    opponent_path = ""  # Path to pre-trained opponent .pth file
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_save_path = None  # Will be set in save_best_fn

def get_env():
    from environment import risk
    raw_env = risk.env()

    return PettingZooEnv(raw_env)

if __name__ == "__main__":


    args = Args()

    train_envs = ts.env.DummyVectorEnv([get_env for _ in range(args.num_train_envs)])
    test_envs = ts.env.DummyVectorEnv([get_env for _ in range(args.num_test_envs)])

    temp_env = get_env()
    agents = temp_env.agents

    obs_space = temp_env.observation_space
    action_space = temp_env.action_space

    net_dqn = ResidualNetwork(
        obs_space=obs_space,
        action_space=action_space,
        hidden_dim=256
    ).to(args.device)
    policy = DiscreteQLearningPolicy(
        model=net_dqn,
        action_space=temp_env.action_space,
        eps_training=args.eps_train,
        eps_inference=args.eps_test
    )
    algorithm = ts.algorithm.DQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=args.lr),
        gamma=args.gamma,
        n_step_return_horizon=args.n_step,
        target_update_freq=args.target_update_freq
    )

    agents = [algorithm] + ([MARLRandomDiscreteMaskedOffPolicyAlgorithm(action_space=temp_env.action_space)] * (temp_env.num_agents-1))
    marl_algorithm = MultiAgentOffPolicyAlgorithm(algorithms=agents, env=temp_env)

    training_collector = Collector[CollectStats](
        marl_algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector[CollectStats](marl_algorithm, test_envs, exploration_noise=True)

    training_collector.reset()
    training_collector.collect(n_step=args.batch_size * args.num_train_envs)

    def save_best_fn(policy: ts.algorithm.Algorithm) -> None:
        if hasattr(args, "model_save_path") and args.model_save_path:
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(args.logdir, "risk", "dqn", "policy.pth")
        torch.save(policy.get_algorithm(player_agent_id).state_dict(), model_save_path)

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.win_rate

    def reward_metric(rews: np.ndarray) -> np.ndarray:
        return rews[:, args.agent_id - 1]

    result = marl_algorithm.run_training(
        OffPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_num_steps,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            test_step_num_episodes=args.num_test_envs,
            batch_size=args.batch_size,
            stop_fn=stop_fn,
            #save_best_fn=save_best_fn,
            update_step_num_gradient_steps_per_sample=args.update_per_step,
            test_in_training=False,
            multi_agent_return_reduction=reward_metric,
            show_progress=False,
        )
    )

    print(result)