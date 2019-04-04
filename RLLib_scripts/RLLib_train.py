from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import gym

import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo.ppo import PPOAgent
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from RLLib_scripts.RLLibCustomModel import RLLibCustomModel, LightModel

from RLLib_scripts.GridWorldRLLibEnv import GridWorldRLLibEnv


'''
Training script for the GridWorldRLLibEnv multiagent Environment.
'''

parser = argparse.ArgumentParser()
parser.add_argument("--num-iters", type=int, default=20)


if __name__ == "__main__":

    # Here we register the custom models
    ModelCatalog.register_custom_model("my_model", RLLibCustomModel)
    ModelCatalog.register_custom_model("light_model", LightModel)


    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--n_round", type=int, default=50000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--agent_generator", default='random_spread', choices=['random_spread',
                                                                               'random_clusters',
                                                                               'random_static_clusters',
                                                                               'random_static_clusters_single_agent',
                                                                               'random_static_clusters_two_agents',
                                                                               'two_clusters',
                                                                               'static_cluster_spaced'])
    parser.add_argument("--map_size", type=int, default=100)
    parser.add_argument("--name", type=str, default="goal")
    args = parser.parse_args()

    # init the game
    print('Init Env')
    if args.agent_generator == 'static_cluster_spaced':
        args.map_size = 16

    if args.agent_generator == 'random_static_clusters_single_agent'\
            or args.agent_generator == 'random_static_clusters_two_agents':
        args.map_size = 40

    # Specifying observation space and actions space dimensions.
    # TODO: take this as parameter and send it to the corresponding magent/builtin/config file.
    obs_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=1, shape=(31, 31, 6)),
                                                gym.spaces.Box(low=0, high=1, shape=(21,) )))
    act_space = (gym.spaces.Discrete(9))

    # Dict with the different policies to train
    policy_graphs = {
        "ppo_policy_agent_0": (PPOPolicyGraph, obs_space, act_space, {}),
        #"ppo_policy_agent_1": (PPOPolicyGraph, obs_space, act_space, {}),
        # "ppo_policy_agent_1": (PPOPolicyGraph, obs_space, act_space, {}),
        # "ppo_policy_agent_2": (PPOPolicyGraph, obs_space, act_space, {}),
        # "ppo_policy_agent_3": (PPOPolicyGraph, obs_space, act_space, {}),
    }

    # Function mapping agent id to the corrresponding policy
    def policy_mapping_fn(agent_id):
        return "ppo_policy_agent_0"
        # if agent_id == "agent_0":
        #     return "ppo_policy_agent_0"
        # return "ppo_policy_agent_1"

    # Environment configuration
    env_config = {"map_size": args.map_size,
            "agent_generator": args.agent_generator,
            "render": args.render,
            "num_static_blocks": 1
    }

    register_env("gridworld", lambda _: GridWorldRLLibEnv(env_config))


    # PPO Config specification
    config = ppo.DEFAULT_CONFIG.copy()
    # config['model'] = {"use_lstm": True}
    # config['model'] = {"custom_model": "light_model"}
    config['model'] = {"fcnet_hiddens": [32,32]}  # Here we use the default fcnet with modified hidden layers size

    config["num_workers"] = 0
    config["num_cpus_per_worker"] = 32
    config["num_gpus"] = 1
    config["num_gpus_per_worker"] = 1
    config["num_cpus_for_driver"] = 8
    config["num_envs_per_worker"] = 15

    # Config for rendering (Only one environment in parallel or there is a bug with de video.txt file.
    # config["num_workers"] = 0
    # config["num_cpus_per_worker"] = 32
    # config["num_gpus"] = 1
    # config["num_cpus_for_driver"] = 8
    # config["num_envs_per_worker"] = 1

    config['multiagent'] = {"policy_graphs": policy_graphs,
                            "policy_mapping_fn": policy_mapping_fn,
                            "policies_to_train": ["ppo_policy_agent_0",
                                                  "ppo_policy_agent_1"]}  # , "ppo_policy_agent_1", "ppo_policy_agent_2",
    # "ppo_policy_agent_3"]}

    ppo_trainer = PPOAgent(env="gridworld", config=config)

    # To reload policies from a checkpoint
    # ppo_trainer.restore('/home/guillaume/ray_results/PPO_gridworld_2019-04-03_00-23-12g7ri2yzi/checkpoint_2602/checkpoint-2602')


    for i in range(args.n_round + 2):
        print("== Iteration", i, "==")

        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        if i % args.save_every == 0:
            checkpoint = ppo_trainer.save()
            print("checkpoint saved at", checkpoint)


