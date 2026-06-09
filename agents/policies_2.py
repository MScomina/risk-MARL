import numpy as np
import os
import torch
import tianshou as ts
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils.net.discrete import DiscreteActor
from tianshou.utils.net.discrete import DiscreteCritic
from tianshou.algorithm.modelfree.reinforce import DiscreteActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.algorithm.random import MARLRandomDiscreteMaskedOffPolicyAlgorithm
from tianshou.algorithm import PPO
from tianshou.env import PettingZooEnv
from tianshou.algorithm.multiagent.marl import MultiAgentOnPolicyAlgorithm
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.trainer import OnPolicyTrainerParams

from models import GraphNetwork

class MARLRandomDiscreteMaskedOnPolicyAdapter(MARLRandomDiscreteMaskedOffPolicyAlgorithm):

    def _update_with_batch(self, batch, batch_size=None, repeat=None, **kwargs):
        return super()._update_with_batch(batch)

class Args:
    seed = 1626
    buffer_size = 1e5
    lr = 1e-4
    gamma = 0.995
    gae_lambda = 0.97
    vf_coef = 0.5
    ent_coef = 5e-3
    dual_clip: float | None = None
    value_clip: bool = True
    advantage_normalization: bool = True
    recompute_advantage: bool = True
    n_step = 512
    target_update_freq = 256
    epoch = 1000
    epoch_start_step = 256
    epoch_max_steps = 16384
    collection_step_num_env_steps = 1024
    update_per_step = 1.0
    batch_size = 256
    num_train_envs = 16
    num_test_envs = 5
    max_grad_norm = 0.3
    eps_clip = 0.2
    logdir = "log"
    render = 0.1
    win_rate = 0.9
    watch = False
    agent_id = 1
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
            "max_iters": min(args.epoch_start_step * (2**(env_length)), args.epoch_max_steps),
            "n_agents": 4,
            "map_path": "./environment/maps/classic.json",
            "is_card_game": True,
            "max_armies": 100
        } for env_length in range(args.num_train_envs)
    ]

    train_envs = ts.env.SubprocVectorEnv([
        (lambda config=param: get_env(config)) for param in env_params
    ])
    
    test_envs = ts.env.SubprocVectorEnv([
        (lambda config=env_params[-1]: get_env(config)) for _ in range(args.num_test_envs)
    ])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    env = get_env(env_params[-1])

    obs_space = env.observation_space.shape
    action_space = env.action_space

    net_actor = GraphNetwork(
        obs_space=env.observation_space,
        action_space=action_space,
        map_graph=env.env.map_network,
        res_hidden_size=128,
        gnn_hidden_size=64,
        residual_depth=2,
        embed_space_owners=8,
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
        gnn_hidden_size=64,
        residual_depth=2,
        embed_space_owners=8,
        embed_space_phase=8,
        activation_function=torch.nn.SiLU,
        starting_residual_scale=0.2,
        n_heads=4
    ).to(args.device)

    policy = DiscreteActorPolicy(
        actor=net_actor,
        action_space=action_space
    )

    algorithm = PPO(
        policy=policy,
        critic=net_critic,
        optim=AdamOptimizerFactory(lr=args.lr),
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        advantage_normalization=args.advantage_normalization,
        recompute_advantage=args.recompute_advantage,
        eps_clip=args.eps_clip
    ).to(args.device)

    full_path = os.path.join(args.logdir, "ppo")
    if args.load_model:
        algorithm.load_state_dict(torch.load(os.path.join(full_path, "actor.pth")))
        net_critic.load_state_dict(torch.load(os.path.join(full_path, "critic.pth")))

    agents = [algorithm] + [
        MARLRandomDiscreteMaskedOnPolicyAdapter(action_space=action_space)
        for _ in range(env.num_agents - 1)
    ]

    marl_algorithm = MultiAgentOnPolicyAlgorithm(algorithms=agents, env=env)

    training_collector = Collector[CollectStats](
        marl_algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector[CollectStats](marl_algorithm, test_envs, exploration_noise=False)

    training_collector.reset()
    training_collector.collect(n_step=args.batch_size * args.num_train_envs)

    target_agent_name = env.agents[0] 
    target_agent_index = env.agents.index(target_agent_name)

    def reward_metric(rews: np.ndarray) -> np.ndarray:
        return rews[:, target_agent_index].flatten()

    def save_best_fn(policy: ts.algorithm.Algorithm) -> None:
        if hasattr(args, "model_save_path") and args.model_save_path:
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(args.logdir, "ppo")
        torch.save(policy.get_algorithm(target_agent_name).state_dict(), os.path.join(model_save_path, "actor.pth"))
        torch.save(net_critic.state_dict(), os.path.join(model_save_path, "critic.pth"))

    log_path = os.path.join(args.logdir, 'ppo')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def train_fn(epoch: int, env_step: int) -> None:
        for name, param in net_actor.named_parameters():
            writer.add_histogram(f"params_act/{name}", param.clone().detach().cpu(), global_step=env_step)
            if param.grad is not None:
                writer.add_histogram(f"grads_act/{name}", param.grad.clone().detach().cpu(), global_step=env_step)
        for name, param in net_critic.named_parameters():
            writer.add_histogram(f"params_crit/{name}", param.clone().detach().cpu(), global_step=env_step)
            if param.grad is not None:
                writer.add_histogram(f"grads_crit/{name}", param.grad.clone().detach().cpu(), global_step=env_step)

        new_ent = args.ent_coef - (args.ent_coef - 0.001) * min(1.0, epoch / (args.epoch))
        algorithm.ent_coef

    result = marl_algorithm.run_training(
        OnPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=args.epoch,
            epoch_num_steps=args.epoch_max_steps,
            update_step_num_repetitions=4,
            test_step_num_episodes=args.num_test_envs,
            batch_size=args.batch_size,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            training_fn=train_fn,
            #stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_training=False,
            multi_agent_return_reduction=reward_metric,
            #resume_from_log=resume_id is not None,
            #save_checkpoint_fn=save_checkpoint_fn,
        )
    )

    print(result)

if __name__ == "__main__":
    main()