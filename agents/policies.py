# https://tianshou.org/en/stable/index.html
# https://tianshou.org/en/stable/02_deep_dives/L6_MARL.html

import os
import math
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

from models import GraphNetwork

class Args:
    seed = 1626
    eps_test = 0.05
    eps_train = 0.9
    buffer_size = 2e6
    lr = 3e-4
    gamma = 0.993
    n_step = 7
    target_update_freq = 1024
    epoch = 500
    epoch_start_step = 64
    epoch_max_steps = 4096
    collection_step_num_env_steps = 2048
    update_per_step = 0.25
    batch_size = 512
    num_train_envs = 16
    num_test_envs = 8
    logdir = "log"
    render = 0.1
    win_rate = 0.9
    watch = False
    agent_id = 1
    resume_path = ""
    opponent_path = ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_model = True
    model_save_path = None

args = Args()

def get_env(env_config):
    from environment import risk
    raw_env = risk.env(
        n_agents=env_config["n_agents"],
        max_iters=env_config["max_iters"],
        map_path=env_config["map_path"],
        is_card_game=env_config["is_card_game"],
        max_armies=env_config["max_armies"]
    )
    return PettingZooEnv(raw_env)

def main():

    env_params = [
        {
            "max_iters": min(args.epoch_start_step * (2**(env_length//2)), args.epoch_max_steps),
            "n_agents": 3,
            "map_path": "./environment/maps/africa-europe.json",
            "is_card_game": True,
            "max_armies": 50
        } for env_length in range(args.num_train_envs)
    ]

    train_envs = ts.env.SubprocVectorEnv([
        (lambda config=param: get_env(config)) for param in env_params
    ])
    
    test_envs = ts.env.SubprocVectorEnv([
        (lambda: get_env(env_params[-1])) for _ in range(args.num_test_envs)
    ])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    temp_env = get_env(env_params[-1])
    agents = temp_env.agents

    obs_space = temp_env.observation_space
    action_space = temp_env.action_space

    net_dqn = GraphNetwork(
        obs_space=obs_space,
        action_space=action_space,
        map_graph=temp_env.env.map_network,
        res_hidden_size=128,
        gnn_hidden_size=64,
        residual_depth=2,
        embed_space_owners=8,
        embed_space_phase=8,
        activation_function=torch.nn.SiLU,
        starting_residual_scale=0.2,
        n_heads=4
    ).to(args.device)


    policy = DiscreteQLearningPolicy(
        model=net_dqn,
        observation_space=temp_env.observation_space,
        action_space=temp_env.action_space,
        eps_training=args.eps_train,
        eps_inference=args.eps_test
    )
    algorithm = ts.algorithm.DQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=args.lr, eps=1e-4),
        gamma=args.gamma,
        n_step_return_horizon=args.n_step,
        target_update_freq=args.target_update_freq
    )
    full_path = os.path.join(args.logdir, "dqn")
    if args.load_model:
        algorithm.load_state_dict(torch.load(os.path.join(full_path, "policy.pth")))

    agents = [algorithm] + [
        MARLRandomDiscreteMaskedOffPolicyAlgorithm(action_space=temp_env.action_space) 
        for _ in range(temp_env.num_agents - 1)
    ]
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

    target_agent_name = temp_env.agents[args.agent_id-1] 

    def save_best_fn(policy: ts.algorithm.Algorithm) -> None:
        if hasattr(args, "model_save_path") and args.model_save_path:
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(args.logdir, "dqn", "policy.pth")
        torch.save(policy.get_algorithm(target_agent_name).state_dict(), model_save_path)

    def stop_fn(mean_rewards: float) -> bool:
        return mean_rewards >= args.win_rate

    print(f"Environment agents: {temp_env.agents}")
    
    target_agent_index = temp_env.agents.index(target_agent_name)
    print(f"Tracking {target_agent_name} at index {target_agent_index}")

    def reward_metric(rews: np.ndarray) -> np.ndarray:
        return rews[:, target_agent_index].flatten()

    log_path = os.path.join(args.logdir, 'dqn')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)


    def get_epsilon(step, start_eps=args.eps_train, end_eps=0.05, decay_steps=args.epoch_max_steps*args.epoch):

        decay_rate = math.exp(math.log(end_eps / start_eps) / decay_steps)
        
        eps = max(end_eps, start_eps * (decay_rate ** step))
        return eps

    def train_fn(epoch: int, env_step: int) -> None:
        for name, param in net_dqn.named_parameters():
            writer.add_histogram(f"params/{name}", param.clone().detach().cpu(), global_step=env_step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad.clone().detach().cpu(), global_step=env_step)
                    
        current_eps = get_epsilon(env_step)
        marl_algorithm.get_algorithm(target_agent_name).policy.set_eps_training(current_eps)

    result = marl_algorithm.run_training(
        OffPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=args.epoch-1,
            epoch_num_steps=args.epoch_max_steps,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            test_step_num_episodes=args.num_test_envs,
            batch_size=args.batch_size,
            stop_fn=stop_fn,
            logger=logger,
            save_best_fn=save_best_fn,
            update_step_num_gradient_steps_per_sample=args.update_per_step,
            test_in_training=False,
            multi_agent_return_reduction=reward_metric,
            show_progress=False,
            training_fn=train_fn
        )
    )

    print(result)

if __name__ == "__main__":
    main()
