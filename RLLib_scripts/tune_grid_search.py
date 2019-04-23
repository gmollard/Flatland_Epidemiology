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

import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.agents.ppo.ppo import PPOAgent
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph

import ray.rllib.agents.dqn as dqn
from ray.rllib.agents.dqn.dqn import DQNAgent
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
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
    obs_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=1, shape=(31, 31, 5)), gym.spaces.Space((21,))))
    #gym.spaces.Box(low=0, high=1, shape=(21,))))
    # obs_space = gym.spaces.Space((31*31*5,))
    act_space = (gym.spaces.Discrete(9))

    policy_graphs = {}
    # Dict with the different policies to train
    if config['independent_training'] == "common_trainer_common_policy":
        policy_graphs[f"ppo_policy_agent_0_independent_training_{config['independent_training']}"] = (PPOPolicyGraph, obs_space, act_space, {})

    else:
        for i in range(config['n_agents']):
            policy_graphs[f"ppo_policy_agent_{i}_independent_training_{config['independent_training']}"]\
                = (PPOPolicyGraph, obs_space, act_space, {})


    def policy_mapping_fn(agent_id):
        # id = int(agent_id.split('_')[-1])
        if config['independent_training'] == "common_trainer_common_policy":
            return f"ppo_policy_agent_0_independent_training_{config['independent_training']}"
        else:
            return f"ppo_policy_{agent_id}_independent_training_{config['independent_training']}"

    # Environment configuration
    env_config = {"map_size": config['map_size'],
                  "agent_generator": 'random_static_clusters_1_to_4_agents',
                  "render": False,
                  "num_static_blocks": 1,
                  "n_agents": config["n_agents"],
                  "vaccine_reward": config["vaccine_reward"],
                  "view_mode": config["view_mode"]
                  }

    register_env(f"gridworld_{config['n_agents']}_independent_training_{config['independent_training']}",
                 lambda _: GridWorldRLLibEnv(env_config))

    # PPO Config specification
    agent_config = ppo.DEFAULT_CONFIG.copy()
    # Here we use the default fcnet with modified hidden layers size

    agent_config['model'] = {"fcnet_hiddens": config['hidden_sizes']}

    agent_config["num_workers"] = 0
    agent_config["num_cpus_per_worker"] = 15
    agent_config["num_gpus"] = 0.5
    agent_config["num_gpus_per_worker"] = 0.5
    agent_config["num_cpus_for_driver"] = 1
    agent_config["num_envs_per_worker"] = 10
    agent_config["vf_clip_param"] = config['vf_clip_param']

    if config['independent_training'] != "independent":
        agent_config['multiagent'] = {"policy_graphs": policy_graphs,
                                "policy_mapping_fn": policy_mapping_fn,
                                "policies_to_train": list(policy_graphs.keys())}
    else:
        agent_config['multiagent'] = {"policy_graphs": policy_graphs,
                                      "policy_mapping_fn": policy_mapping_fn,
                                      "policies_to_train": [list(policy_graphs.keys())[0]]}

    def logger_creator(conf):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        logdir = f"independent_training_{config['independent_training']}"
        logdir = tempfile.mkdtemp(
            prefix=logdir, dir="/mount/SDC/independent_ppo_training")
        return UnifiedLogger(conf, logdir, None)

    logger = logger_creator

    ppo_trainer = PPOAgent(env=f"gridworld_{config['n_agents']}_independent_training_{config['independent_training']}", config=agent_config, logger_creator=logger)

    if config['independent_training'] == "independent":
        agent_config_2 = ppo.DEFAULT_CONFIG.copy()
        agent_config_2['model'] = {"fcnet_hiddens": config['hidden_sizes']}

        agent_config_2["num_workers"] = 0
        agent_config_2["num_cpus_per_worker"] = 15
        agent_config_2["num_gpus"] = 0.5
        agent_config_2["num_gpus_per_worker"] = 0.5
        agent_config_2["num_cpus_for_driver"] = 1
        agent_config_2["num_envs_per_worker"] = 10
        agent_config_2["vf_clip_param"] = config['vf_clip_param']


        agent_config_2['multiagent'] = {"policy_graphs": policy_graphs,
                                      "policy_mapping_fn": policy_mapping_fn,
                                      "policies_to_train": [list(policy_graphs.keys())[1]]}

        ppo_trainer_2 = PPOAgent(env=f"gridworld_{config['n_agents']}_independent_training_{config['independent_training']}", config=agent_config_2, logger_creator=logger)


    # ppo_trainer.restore('/mount/SDC/ray_results/1_agents_single_policy_Falsenjj_o0sl/checkpoint_1401/checkpoint-1401')
    for i in range(100000 + 2):
        print("== Iteration", i, "==")

        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        if config['independent_training'] == "independent":
            # swap weights to synchronize
            # ppo_trainer.set_weights(ppo_trainer_2.get_weights([f"ppo_policy_1_independent_training_{config['independent_training']}"]))
            ppo_trainer_2.set_weights(ppo_trainer.get_weights([f"ppo_policy_0_independent_training_{config['independent_training']}"]))

        if config['independent_training'] == "independent":
            print(pretty_print(ppo_trainer_2.train()))
            if i % config['save_every'] == 0:
                checkpoint = ppo_trainer_2.save()
                print("checkpoint saved at", checkpoint)

        if i % config['save_every'] == 0:
            checkpoint = ppo_trainer.save()
            print("checkpoint saved at", checkpoint)

        if config['independent_training'] == "independent":
            # swap weights to synchronize
            ppo_trainer.set_weights(ppo_trainer_2.get_weights([f"ppo_policy_1_independent_training_{config['independent_training']}"]))
            # ppo_trainer_2.set_weights(ppo_trainer.get_weights([f"ppo_policy_0_independent_training_{config['independent_training']}"]))

        reporter(num_iterations_trained=ppo_trainer._iteration)


all_trials = tune.run(
    train_func,
    name="observation_tuned",
    stop={"num_iterations_trained": 3202},
    config={"single_policy": False,
            "n_agents": 2,
            "hidden_sizes": [128, 128],
            "save_every": 200,
            "map_size": 40,
            "vaccine_reward": 0.1,
            "vf_clip_param": 10,
            "view_mode": "dist_map",
            "independent_training": tune.grid_search(["independent", "common_trainer", "common_trainer_common_policy"])
            # tune.grid_search([10, 100, 1000])#, 0, 0.1])
            # "view_mode": "exposed_agents_immunized"
            },
    resources_per_trial={
        "cpu": 15,
        "gpu": 0.5
    },
    local_dir="/mount/SDC/independent_ppo_training"
)
