from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""Example of using two different training methods at once in multi-agent.
Here we create a number of CartPole agents, some of which are trained with
DQN, and some of which are trained with PPO. We periodically sync weights
between the two trainers (note that no such syncing is needed when using just
a single training method).
For a simpler example, see also: multiagent_cartpole.py
"""

import argparse

import gym

import ray
from ray.rllib.agents.dqn.dqn import DQNAgent
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.agents.ppo.ppo import PPOAgent
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog, Model
from examples.RLLibCustomModel import RLLibCustomModel


from magent.GridWorldRLLibEnv import GridWorldRLLibEnv

parser = argparse.ArgumentParser()
parser.add_argument("--num-iters", type=int, default=20)


if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("my_model", RLLibCustomModel)

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--n_round", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--agent_generator", default='random_spread', choices=['random_spread',
                                                                               'random_clusters',
                                                                               'random_static_clusters',
                                                                               'two_clusters',
                                                                               'static_cluster_spaced'])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--map_size", type=int, default=80)
    parser.add_argument("--name", type=str, default="goal")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'pytorch_dqn', 'drqn', 'a2c'])
    args = parser.parse_args()

    # init the game
    print('Init Env')
    if args.agent_generator == 'static_cluster_spaced':
        args.map_size = 16

    if args.agent_generator == 'random_static_clusters':
        args.map_size = 40  # 55


    obs_space = gym.spaces.Tuple((gym.spaces.Space((31,31,6)), gym.spaces.Space((21,))))
    act_space = gym.spaces.Discrete(9)


    policy_graphs = {
        "dqn_policy_agent_0": (PPOPolicyGraph, obs_space, act_space, {'gamma': 0.99}),
        "dqn_policy_agent_1": (PPOPolicyGraph, obs_space, act_space, {'gamma': 0.95}),
        "dqn_policy_agent_2": (PPOPolicyGraph, obs_space, act_space, {'gamma': 0.90}),
        "dqn_policy_agent_3": (PPOPolicyGraph, obs_space, act_space, {'gamma': 0.85})
    }

    def policy_mapping_fn(agent_id):
        return f"dqn_policy_{agent_id}"

    env_config = {"map_size": args.map_size,
            "agent_generator": args.agent_generator,
            "render": args.render
    }

    register_env("gridworld", lambda _: GridWorldRLLibEnv(env_config))

    dqn_trainer = PPOAgent(
        env="gridworld",
        config={
            "multiagent": {
                "policy_graphs": policy_graphs,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["dqn_policy"],
            },
            "model": {
                "custom_model": "my_model",
            },
            "num_workers":16,
            "num_gpus": 2,
            "num_envs_per_worker": 2
            # "gamma": 0.95,
            # "n_step": 3,
        })


    for i in range(args.n_round):
        print("== Iteration", i, "==")

        # improve the DQN policy
        print("-- DQN --")
        print(pretty_print(dqn_trainer.train()))

        if i % 20 == 0:
            checkpoint = dqn_trainer.save()
            print("checkpoint saved at", checkpoint)


