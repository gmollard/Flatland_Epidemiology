from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pprint import pprint
import os

import argparse

import gym
import tempfile
import ray
from ray.tune.logger import UnifiedLogger

import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo.ppo import PPOAgent
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from RLLib_scripts.RLLibCustomModel import RLLibCustomModel, LightModel

from RLLib_scripts.GridWorldRLLibEnv import GridWorldRLLibEnv
from ray import tune

ray.init()

def train_func(config, reporter):
    # init the game
    print('Init Env')

    # Specifying observation space and actions space dimensions.
    # TODO: take this as parameter and send it to the corresponding magent/builtin/config file.
    obs_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=1, shape=(31, 31, 6)),
                                  gym.spaces.Box(low=0, high=1, shape=(21,))))
    act_space = (gym.spaces.Discrete(9))

    policy_graphs = {}
    # Dict with the different policies to train
    if config['single_policy']:
        policy_graphs[f"ppo_policy_agent_0_{config['n_agents']}1_{config['single_policy']}"] = (PPOPolicyGraph, obs_space, act_space, {})

    else:
        for i in range(config['n_agents']):
            policy_graphs[f"ppo_policy_agent_{i}_1_False"] = (PPOPolicyGraph, obs_space, act_space, {})


    def policy_mapping_fn(agent_id):
        # id = int(agent_id.split('_')[-1])
        if config['single_policy']:
            return f"ppo_policy_agent_0_{config['n_agents']}_{config['single_policy']}"
        else:
            return f"ppo_policy_{agent_id}_1_False"


    # Environment configuration
    env_config = {"map_size": config['map_size'],
                  "agent_generator": 'random_static_clusters_1_to_4_agents',
                  "render": True,
                  "num_static_blocks": 1,
                  "vaccine_reward": 0.1,
                  "n_agents": config["n_agents"]
                  }
        # if agent_id == "agent_

    register_env(f"gridworld_{config['n_agents']}", lambda _: GridWorldRLLibEnv(env_config))

    # PPO Config specification
    agent_config = ppo.DEFAULT_CONFIG.copy()
    # Here we use the default fcnet with modified hidden layers size
    agent_config['model'] = {"fcnet_hiddens": config['hidden_sizes']}

    agent_config["num_workers"] = 0
    agent_config["num_cpus_per_worker"] = 40
    agent_config["num_gpus"] = 2
    agent_config["num_gpus_per_worker"] = 2
    agent_config["num_cpus_for_driver"] = 5
    agent_config["num_envs_per_worker"] = 1

    agent_config['multiagent'] = {"policy_graphs": policy_graphs,
                            "policy_mapping_fn": policy_mapping_fn,
                            "policies_to_train": list(policy_graphs.keys())}

    def logger_creator(conf):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        logdir = f"single_agent"
        logdir = tempfile.mkdtemp(
            prefix=logdir, dir="/mount/SDC/ray_results_single_agent")
        return UnifiedLogger(conf, logdir, None)

    logger = logger_creator

    ppo_trainer = PPOAgent(env=f"gridworld_{config['n_agents']}", config=agent_config, logger_creator=logger)

    # ppo_trainer.restore(f'/mount/SDC/ray_results_single_agent/single_agentj2rdenlx/checkpoint_2802/checkpoint-2802')

    for i in range(100000 + 2):
        print("== Iteration", i, "==")

        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        if i % config['save_every'] == 0:
            checkpoint = ppo_trainer.save()
            print("checkpoint saved at", checkpoint)

        reporter(num_iterations_trained=ppo_trainer._iteration)


all_trials = tune.run(
    train_func,
    name="n_agents_policy_grid_search",
    stop={"num_iterations_trained": 50000},
    config={"single_policy": False,
            "map_size": 40,
            "n_agents": 4,
            "hidden_sizes": [32, 32],
            "save_every": 200},
            # "checkpoint_path": tune.grid_search(['grid_sizes_60uiqrcefy', 'grid_sizes_70yogg45vy', 'grid_sizes_80aezp1jc3'])},
    resources_per_trial={
        "cpu": 45,
        "gpu": 2
    },
    # local_dir="/mount/SDC/ray_results_single_agent"
)
