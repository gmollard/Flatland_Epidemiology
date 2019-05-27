from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tempfile
import ray
from ray.tune.logger import UnifiedLogger

import ray.rllib.agents.ppo.ppo as ppo
import ray.rllib.agents.dqn.dqn as dqn
# from ray.rllib.agents.ppo.ppo import PPOTrainer
from RLLib_scripts.ppo import PPOTrainer
from ray.rllib.agents.dqn.dqn import DQNTrainer
from RLLib_scripts.ppo_policy_graph_centralized_2 import PPOPolicyGraph
# from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.rllib.models import ModelCatalog

from RLLib_scripts.RLLibCustomModel import LightModel

from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from RLLib_scripts.GridWorldRLLibEnv import GridWorldRLLibEnv
from ray import tune
import gin

from ray.rllib.models.preprocessors import Preprocessor
import numpy as np

class MyPreprocessorClass(Preprocessor):
    def _init_shape(self, obs_space, options):
        return (4*17**2 + 21,)

    def transform(self, observation):
        return np.concatenate([observation[0].flatten(), observation[1]])  # return the preprocessed observation

ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)
ray.init()
# ray.init(object_store_memory=150000000000)
ModelCatalog.register_custom_model("conv_model", LightModel)


def train_func(config):

    # init the game
    print('Init Env')
    policy_name = config['policy_name'].format(**locals())#.replace('.','')

    # Specifying observation space and actions space dimensions.
    view_radius = config["view_radius"]*2 + 1
    obs_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=1, shape=(view_radius, view_radius, 4)),
                                  gym.spaces.Space((21,))))
    act_space = (gym.spaces.Discrete(9))

    policy_graphs = {}
    # Dict with the different policies to train
    policy_graphs[policy_name] = (PPOPolicyGraph, obs_space, act_space, {})
    # policy_graphs["ppo_policy_infection_prob_003"] = (PPOPolicyGraph, obs_space, act_space, {})

    def policy_mapping_fn(agent_id):
        return policy_name

    # Environment configuration
    env_config = {"map_size": config['map_size'],
                  "agent_generator": 'large_toric_env',
                  "render": False,
                  "num_static_blocks": 1,
                  "n_agents": config["n_agents"],
                  "vaccine_reward": config["vaccine_reward"],
                  "view_radius": config["view_radius"],
                  "step_reward": config["step_reward"],
                  "final_reward": config["final_reward"],
                  "final_reward_times_healthy": config["final_reward_times_healthy"],
                  "bad_vaccine_penalty": -0.1,
                  "collide_penalty": -0.1,
                  "horizon": config["horizon"],
                  "infection_prob": config["infection_prob"],
                  "initially_infected": config["initially_infected"],
                  "decreasing_vaccine_reward": config["decreasing_vaccine_reward"]
                  }

    # PPO Config specification
    agent_config = ppo.DEFAULT_CONFIG.copy()
    # Here we use the default fcnet with modified hidden layers size
    agent_config['env_config'] = env_config
    agent_config['model'] = {"fcnet_hiddens": config['hidden_sizes'], "custom_preprocessor": "my_prep"}
    #agent_config['model'] = {"custom_model": "conv_model"}

    agent_config["num_workers"] = 0
    agent_config["num_cpus_per_worker"] = 2
    agent_config["num_gpus"] = 0
    agent_config["num_gpus_per_worker"] = 0
    agent_config["num_cpus_for_driver"] = 1
    agent_config["num_envs_per_worker"] = 1
    agent_config["batch_mode"] = "complete_episodes"
    agent_config["vf_clip_param"] = config['vf_clip_param']
    agent_config["vf_share_layers"] = config['vf_share_layers']
    agent_config["simple_optimizer"] = False
    agent_config["entropy_coeff"] = config["entropy_coeff"]
    agent_config['use_centralized_vf'] = True
    agent_config['max_vf_agents'] = 20
    # agent_config["n_step"] = 3

    agent_config['multiagent'] = {"policy_graphs": policy_graphs,
                                "policy_mapping_fn": policy_mapping_fn,
                                "policies_to_train": list(policy_graphs.keys())}


    def logger_creator(conf):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        logdir = policy_name
        logdir = tempfile.mkdtemp(
            prefix=logdir, dir=config['local_dir'])
        return UnifiedLogger(conf, logdir, None)

    logger = logger_creator

    ppo_trainer = PPOTrainer(env=GridWorldRLLibEnv, config=agent_config, logger_creator=logger)

    # ppo_trainer.restore('/home/guillaume/Desktop/distMAgent/large_toric_env_negative_vaccine_reward/ppo_policy_vaccine_reward_0.01_times_healthy_Truej1lg3swm/checkpoint_6201/checkpoint-6201')

    for i in range(100000 + 2):
        print("== Iteration", i, "==")

        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        # if i % config['save_every'] == 0:
        #     checkpoint = ppo_trainer.save()
        #     print("checkpoint saved at", checkpoint)



@gin.configurable
def run_grid_search(name, view_radius, n_agents, hidden_sizes, save_every, map_size, vaccine_reward,
                vf_clip_param, num_iterations, vf_share_layers, step_reward, final_reward,
                    final_reward_times_healthy, entropy_coeff, local_dir, learning_rate, policy_name, infection_prob,
                    initially_infected, decreasing_vaccine_reward, horizon=False):

    # tune.run(
    #     train_func,
    #     name=name,
    #     stop={"num_iterations_trained": num_iterations},
    #     config={"view_radius": view_radius,
    #             "n_agents": n_agents,
    #             "hidden_sizes": hidden_sizes,
    #             "save_every": save_every,
    #             "map_size": map_size,
    #             "vaccine_reward": vaccine_reward,
    #             "vf_clip_param": vf_clip_param,
    #             "vf_share_layers": vf_share_layers,
    #             "step_reward": step_reward,
    #             "final_reward": final_reward,
    #             "final_reward_times_healthy": final_reward_times_healthy,
    #             "entropy_coeff": entropy_coeff,
    #             "local_dir": local_dir,
    #             "horizon": horizon
    #             },
    #     resources_per_trial={
    #         "cpu": 11,
    #         "gpu": 0.25
    #     },
    #     local_dir=local_dir
    config = {"view_radius": view_radius,
              "n_agents": n_agents,
              "hidden_sizes": hidden_sizes,
              "save_every": save_every,
              "map_size": map_size,
              "vaccine_reward": vaccine_reward,
              "vf_clip_param": vf_clip_param,
              "vf_share_layers": vf_share_layers,
              "step_reward": step_reward,
              "final_reward": final_reward,
              "final_reward_times_healthy": True, #final_reward_times_healthy,
              "entropy_coeff": 1e-2,#entropy_coeff,
              "local_dir": local_dir,
              "horizon": horizon,
              "policy_name": policy_name,
              "infection_prob": infection_prob,
              "initially_infected": initially_infected,
              "decreasing_vaccine_reward": decreasing_vaccine_reward
                          }
    train_func(config)




if __name__ == '__main__':
    gin.external_configurable(tune.grid_search)
    dir = '/home/guillaume/Desktop/distMAgent/large_toric_env_negative_vaccine_reward'
    gin.parse_config_file(dir + '/config.gin')
    run_grid_search(local_dir=dir)

