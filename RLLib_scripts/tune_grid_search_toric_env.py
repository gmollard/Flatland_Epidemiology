from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tempfile
import ray
from ray.tune.logger import UnifiedLogger

import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.agents.ppo.ppo import PPOAgent
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from RLLib_scripts.GridWorldRLLibEnv import GridWorldRLLibEnv
from ray import tune
import gin

ray.init()


def train_func(config, reporter):
    # init the game
    print('Init Env')

    # Specifying observation space and actions space dimensions.
    view_radius = config["view_radius"]*2 + 1
    obs_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=1, shape=(view_radius, view_radius, 4)),
                                  gym.spaces.Space((21,))))
    act_space = (gym.spaces.Discrete(9))

    policy_graphs = {}
    # Dict with the different policies to train
    policy_graphs[f"ppo_policy_agent_0_vaccine_reward{str(config['vaccine_reward']).replace('.', '')}"] =\
            (PPOPolicyGraph, obs_space, act_space, {})
    # else:
    #     for i in range(config['n_agents']):
    #         policy_graphs[f"ppo_policy_agent_{i}_independent_training_{config['independent_training']}"]\
    #             = (PPOPolicyGraph, obs_space, act_space, {})

    def policy_mapping_fn(agent_id):
        return f"ppo_policy_agent_0_vaccine_reward{str(config['vaccine_reward']).replace('.', '')}"
        # if config['independent_training'] == "common_trainer_common_policy":
        #     return f"ppo_policy_agent_0_vaccine_reward{str(config['vaccine_reward']).replace('.', '')}"
        # else:
        #     return f"ppo_policy_{agent_id}_independent_training_{config['independent_training']}"

    # Environment configuration
    env_config = {"map_size": config['map_size'],
                  "agent_generator": 'toric_env',
                  "render": True,
                  "num_static_blocks": 1,
                  "n_agents": config["n_agents"],
                  "vaccine_reward": config["vaccine_reward"],
                  "view_radius": config["view_radius"],
                  }

    register_env(f"gridworld_{config['n_agents']}",
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
    agent_config["num_envs_per_worker"] = 1
    agent_config["vf_clip_param"] = config['vf_clip_param']

    agent_config['multiagent'] = {"policy_graphs": policy_graphs,
                                "policy_mapping_fn": policy_mapping_fn,
                                "policies_to_train": list(policy_graphs.keys())}

    def logger_creator(conf):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        logdir = f"toric_env_reward_{str(config['vaccine_reward']).replace('.', '')}"
        logdir = tempfile.mkdtemp(
            prefix=logdir, dir=config['local_dir'])
        return UnifiedLogger(conf, logdir, None)

    logger = logger_creator

    ppo_trainer = PPOAgent(env=f"gridworld_{config['n_agents']}", config=agent_config, logger_creator=logger)

    # ppo_trainer.restore('/mount/SDC/ray_results/
    # 1_agents_single_policy_Falsenjj_o0sl/checkpoint_1401/checkpoint-1401')

    for i in range(100000 + 2):
        print("== Iteration", i, "==")

        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        if i % config['save_every'] == 0:
            checkpoint = ppo_trainer.save()
            print("checkpoint saved at", checkpoint)

        reporter(num_iterations_trained=ppo_trainer._iteration)


@gin.configurable
def run_grid_search(name, view_radius, n_agents, hidden_sizes, save_every, map_size, vaccine_reward,
                vf_clip_param, num_iterations, local_dir):

    tune.run(
        train_func,
        name=name,
        stop={"num_iterations_trained": num_iterations},
        config={"view_radius": view_radius,
                "n_agents": n_agents,
                "hidden_sizes": hidden_sizes,
                "save_every": save_every,
                "map_size": map_size,
                "vaccine_reward": vaccine_reward,
                "vf_clip_param": vf_clip_param,
                "local_dir": local_dir
                },
        resources_per_trial={
            "cpu": 45,
            "gpu": 0.5
        },
        local_dir=local_dir
    )




if __name__ == '__main__':
    gin.external_configurable(tune.grid_search)
    dir = '/mount/SDC/toric_env_grid_searches/first_test'
    gin.parse_config_file(dir + '/config.gin')
    run_grid_search(local_dir=dir)

