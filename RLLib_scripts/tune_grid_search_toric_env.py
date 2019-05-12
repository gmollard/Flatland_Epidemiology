from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tempfile
import ray
from ray.tune.logger import UnifiedLogger

import ray.rllib.agents.ppo.ppo as ppo
import ray.rllib.agents.dqn.dqn as dqn
from ray.rllib.agents.ppo.ppo import PPOAgent
from ray.rllib.agents.dqn.dqn import DQNAgent
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
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
        #print(np.concatenate([observation[0].flatten(), observation[1]]).shape)
        #return observation[0]
        return np.concatenate([observation[0].flatten(), observation[1]])  # return the preprocessed observation

ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)
ray.init(object_store_memory=150000000000)
ModelCatalog.register_custom_model("conv_model", LightModel)

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
    policy_graphs[f"ppo_policy_horizon_{str(config['learning_rate']).replace('.', '')}"] =\
            (PPOPolicyGraph, obs_space, act_space, {})
    # else:
    #     for i in range(config['n_agents']):
    #         policy_graphs[f"ppo_policy_agent_{i}_independent_training_{config['independent_training']}"]\
    #             = (PPOPolicyGraph, obs_space, act_space, {})

    def policy_mapping_fn(agent_id):
        return f"ppo_policy_horizon_{str(config['learning_rate']).replace('.', '')}"
        # if config['independent_training'] == "common_trainer_common_policy":
        #     return f"ppo_policy_agent_0_vaccine_reward{str(config['vaccine_reward']).replace('.', '')}"
        # else:
        #     return f"ppo_policy_{agent_id}_independent_training_{config['independent_training']}"

    # Environment configuration
    env_config = {"map_size": config['map_size'],
                  "agent_generator": 'toric_env',
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
                  "horizon": config["horizon"]
                  }

    register_env(f"gridworld_horizon_{str(config['learning_rate']).replace('.', '')}",
                 lambda _: GridWorldRLLibEnv(env_config))

    # PPO Config specification
    agent_config = ppo.DEFAULT_CONFIG.copy()
    # Here we use the default fcnet with modified hidden layers size
    #print('DEFAULT_PREPROCESSOR:', agent_config['preprocessor_pref'])
    agent_config['model'] = {"fcnet_hiddens": config['hidden_sizes'], "custom_preprocessor": "my_prep"}
    #agent_config['model'] = {"custom_model": "conv_model", "custom_preprocessor": "my_prep"}

    agent_config["num_workers"] = 0
    agent_config["num_cpus_per_worker"] = 10
    agent_config["num_gpus"] = 0.5
    agent_config["num_gpus_per_worker"] = 0.5
    agent_config["num_cpus_for_driver"] = 1
    agent_config["num_envs_per_worker"] = 10
    agent_config["batch_mode"] = "complete_episodes"
    agent_config["vf_clip_param"] = config['vf_clip_param']
    agent_config["vf_share_layers"] = config['vf_share_layers']
    agent_config["simple_optimizer"] = False
    agent_config["entropy_coeff"] = config["entropy_coeff"]
    # agent_config["n_step"] = 3

    agent_config['multiagent'] = {"policy_graphs": policy_graphs,
                                "policy_mapping_fn": policy_mapping_fn,
                                "policies_to_train": list(policy_graphs.keys())}

    agent_config['lr'] = config['learning_rate']
    agent_config['horizon'] = config['horizon']
    def logger_creator(conf):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        logdir = f"ppo_policy_horizon_{str(config['learning_rate']).replace('.', '')}"
        logdir = tempfile.mkdtemp(
            prefix=logdir, dir=config['local_dir'])
        return UnifiedLogger(conf, logdir, None)

    logger = logger_creator

    ppo_trainer = PPOAgent(env=f"gridworld_horizon_{str(config['learning_rate']).replace('.', '')}",
                           config=agent_config, logger_creator=logger)

    # ppo_trainer.restore('/mount/SDC/toric_env_grid_searches/simple_optimizer_constant_final_reward/ppo_policy_step_reward-001_final_reward_1eamh2814/checkpoint_1001/checkpoint-1001')

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
                vf_clip_param, num_iterations, vf_share_layers, step_reward, final_reward,
                    final_reward_times_healthy, entropy_coeff, local_dir, learning_rate, horizon=False):

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
                "vf_share_layers": vf_share_layers,
                "step_reward": step_reward,
                "final_reward": final_reward,
                "final_reward_times_healthy": final_reward_times_healthy,
                "entropy_coeff": entropy_coeff,
                "local_dir": local_dir,
                "horizon": horizon,
		"learning_rate": learning_rate
                },
        resources_per_trial={
            "cpu": 11,
            "gpu": 0.5
        },
        local_dir=local_dir
    )




if __name__ == '__main__':
    gin.external_configurable(tune.grid_search)
    dir = '/home/guillaume/sdd/toric_env_grid_searches/learning_rate_grid_search'
    gin.parse_config_file(dir + '/config.gin')
    run_grid_search(local_dir=dir)

