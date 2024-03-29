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


import ray

from RLLib_scripts.GridWorldRLLibEnv import GridWorldRLLibEnv



if __name__ == "__main__":
    ray.init(object_store_memory=150000000000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--n_round", type=int, default=50000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # init the game
    print('Init Env')
    map_size = 40

    # Specifying observation space and actions space dimensions.
    # TODO: take this as parameter and send it to the corresponding magent/builtin/config file.
    obs_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=1, shape=(31, 31, 6)),
                                                gym.spaces.Box(low=0, high=1, shape=(21,) )))
    act_space = (gym.spaces.Discrete(9))

    # Dict with the different policies to train
    policy_graphs = {
        "ppo_policy": (PPOPolicyGraph, obs_space, act_space, {}),
        # "ppo_policy_agent_1_vaccine_reward_01_vf_clip_param_10": (PPOPolicyGraph, obs_space, act_space, {}),
        # "ppo_policy_agent_2_vaccine_reward_01_vf_clip_param_10": (PPOPolicyGraph, obs_space, act_space, {}),
        # "ppo_policy_agent_3_vaccine_reward_01_vf_clip_param_10": (PPOPolicyGraph, obs_space, act_space, {}),
        # "ppo_policy_agent_3": (PPOPolicyGraph, obs_space, act_space, {}),
    }

    # Function mapping agent id to the corrresponding policy
    def policy_mapping_fn(agent_id):
        return f"ppo_policy"


    # Environment configuration
    env_config = {"map_size": map_size,
                  "agent_generator": "randomized_init",
                  "render": args.render,
                  "num_static_blocks": 1,
                  "n_agents": [map_size**2 /50, map_size**2 / 20],
                  "vaccine_reward": 0.0
    }

    register_env("gridworld", lambda _: GridWorldRLLibEnv(env_config))


    # PPO Config specification
    config = ppo.DEFAULT_CONFIG.copy()
    config['model'] = {"fcnet_hiddens": [64, 64]}  # Here we u0e the default fcnet with modified hidden layers size

    # config["num_workers"] = 8
    # config["num_cpus_per_worker"] = 5
    # config["num_gpus"] = 2
    # config["num_gpus_per_worker"] = 0.2
    # config["num_cpus_for_driver"] = 5
    # config["num_envs_per_worker"] = 1
    # config['sample_batch_size'] = 200
    # config['train_batch_size'] = 16
    # config['sgd_minibatch_size'] = 4

    # Config for rendering (Only one environment in parallel or there is a bug with de video.txt file.
    # if args.render:
    config["num_workers"] = 0
    config["num_cpus_per_worker"] = 40
    config["num_gpus"] = 2
    config["num_gpus_per_worker"] = 2
    config["num_cpus_for_driver"] = 5
    config["num_envs_per_worker"] = 10 #10

    config['multiagent'] = {"policy_graphs": policy_graphs,
                            "policy_mapping_fn": policy_mapping_fn,
                            "policies_to_train": list(policy_graphs.keys())}
    # "ppo_policy_agent_3"]

    # def logger_creator(conf):
    #     """Creates a Unified logger with a default logdir prefix
    #     containing the agent name and the env id
    #     """
    #     logdir = f"ppo_4_agents_4_policies"
    #     logdir = tempfile.mkdtemp(
    #         prefix=logdir, dir="/mount/SDC/ray_results_long_run")
    #     return UnifiedLogger(conf, logdir, None)

    # logger = logger_creator

    ppo_trainer = PPOAgent(env="gridworld", config=config)

    # To reload policies from a checkpoint
    ppo_trainer.restore('/home/guillaume/ray_results/PPO_gridworld_2019-04-25_21-18-17zpnm2m1v/checkpoint_101/checkpoint-101')

    for i in range(args.n_round + 2):
        print("== Iteration", i, "==")

        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        if i % args.save_every == 0:
            checkpoint = ppo_trainer.save()
            print("checkpoint saved at", checkpoint)
