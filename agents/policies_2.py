# https://github.com/thu-ml/tianshou/blob/master/examples/vizdoom/vizdoom_ppo.py
import numpy as np
import copy
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

from models.models import GraphNetwork

class MARLRandomDiscreteMaskedOnPolicyAdapter(MARLRandomDiscreteMaskedOffPolicyAlgorithm):

    def _update_with_batch(self, batch, batch_size=None, repeat=None, **kwargs):
        return super()._update_with_batch(batch)

class Args:
    # https://tianshou.org/en/stable/03_api/highlevel/params/algorithm_params.html#tianshou.highlevel.params.algorithm_params.PPOParams
    map_name = "classic"
    seed = 1626
    lr = 1e-4
    gamma = 0.995
    gae_lambda = 0.95
    vf_coef = 0.4
    start_ent_coef = 3e-2
    end_ent_coef = 1e-2
    ent_coef_decay_period = 0.5
    dual_clip: float | None = None
    value_clip: bool = True
    advantage_normalization: bool = True
    recompute_advantage: bool = True
    return_scaling : bool = True
    max_epochs = 60
    epoch_num_steps = 2**17
    env_start_steps = 500
    env_max_steps = 4000
    collection_step_num_env_steps = 16384
    update_per_step = 1.0
    batch_size = 512
    num_train_envs = 16
    num_test_envs = 16
    max_grad_norm = 0.5
    eps_clip = 0.2
    logdir = "log"
    render = 0.1
    watch = False
    agent_id = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_model = False
    model_save_path = None
    log_interval_epochs = 1
    update_step_num_repetitions = 2
    buffer_size = collection_step_num_env_steps * 4

args = Args()

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
        gamma=env_config["gamma"]
    )
    return PettingZooEnv(raw_env)

def main():

    env_params = [
        {
            "max_iters": min(args.env_start_steps * (2**(env_length//2)), args.env_max_steps),
            "n_agents": 3,
            "map_path": f"./environment/maps/{args.map_name}.json",
            "is_card_game": True,
            "max_armies": 2000,
            "dense_rewards": True,
            "is_blitz" : True,
            "gamma" : args.gamma

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
        ent_coef=args.start_ent_coef,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        advantage_normalization=args.advantage_normalization,
        recompute_advantage=args.recompute_advantage,
        eps_clip=args.eps_clip,
        return_scaling=args.return_scaling
    ).to(args.device)

    full_path = os.path.join(args.logdir, "ppo")
    if args.load_model:
        path = os.path.join(full_path, "actor.pth")

        state_dict = torch.load(path, map_location=args.device)

        algorithm.load_state_dict(state_dict)

        net_critic.load_state_dict(
            torch.load(os.path.join(full_path, "critic.pth"),
                    map_location=args.device)
        )

    agents = [algorithm] + [
        MARLRandomDiscreteMaskedOnPolicyAdapter(action_space=action_space)
        for _ in range(env.num_agents - 1)
    ]

    marl_algorithm = MultiAgentOnPolicyAlgorithm(algorithms=agents, env=env)

    training_collector = Collector[CollectStats](
        marl_algorithm,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=False,
    )
    test_collector = Collector[CollectStats](marl_algorithm, test_envs, exploration_noise=False)

    training_collector.reset()
    training_collector.collect(n_step=args.batch_size * args.num_train_envs)

    target_agent_name = env.agents[0] 
    target_agent_index = env.agents.index(target_agent_name)

    def reward_metric(rews: np.ndarray) -> np.ndarray:
        return rews[:, target_agent_index].flatten()

    def save_best_fn(policy: ts.algorithm.Algorithm) -> None:
        model_save_path = args.model_save_path or os.path.join(args.logdir, "ppo")
        os.makedirs(model_save_path, exist_ok=True)

        alg = policy.get_algorithm(target_agent_name)

        torch.save(alg.state_dict(), os.path.join(model_save_path, "actor.pth"))
        torch.save(net_critic.state_dict(), os.path.join(model_save_path, "critic.pth"))

    log_path = os.path.join(args.logdir, 'ppo', args.map_name)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def train_fn(epoch: int, env_step: int) -> None:
        if (epoch % args.log_interval_epochs == 0) and (env_step % args.epoch_num_steps == 0):
            for name, param in net_actor.named_parameters():
                writer.add_histogram(f"params_act/{name}", param.clone().detach().cpu(), global_step=env_step)
                if param.grad is not None:
                    writer.add_histogram(f"grads_act/{name}", param.grad.clone().detach().cpu(), global_step=env_step)
            for name, param in net_critic.named_parameters():
                writer.add_histogram(f"params_crit/{name}", param.clone().detach().cpu(), global_step=env_step)
                if param.grad is not None:
                    writer.add_histogram(f"grads_crit/{name}", param.grad.clone().detach().cpu(), global_step=env_step)

        new_ent = args.start_ent_coef - (args.start_ent_coef - args.end_ent_coef) * min(1.0, epoch / (args.max_epochs*args.ent_coef_decay_period))
        algorithm.ent_coef = new_ent

    result = marl_algorithm.run_training(
        OnPolicyTrainerParams(
            training_collector=training_collector,
            test_collector=test_collector,
            max_epochs=args.max_epochs,
            epoch_num_steps=args.epoch_num_steps,
            update_step_num_repetitions=args.update_step_num_repetitions,
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