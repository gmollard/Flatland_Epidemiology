from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tempfile
import ray
from ray.tune.logger import UnifiedLogger
import pickle

import ray.rllib.agents.ppo.ppo as ppo
import ray.rllib.agents.dqn.dqn as dqn
#from ray.rllib.agents.ppo.ppo import PPOTrainer
from RLLib_scripts.ppo import PPOTrainer
from ray.rllib.agents.dqn.dqn import DQNTrainer
#from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from RLLib_scripts.ppo_policy_graph_centralized_vf import PPOPolicyGraph
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
        return (4*obs_space[0].shape[0]**2 + 2,)

    def transform(self, observation):
        #print(np.concatenate([observation[0].flatten(), observation[1]]).shape)
        #return observation[0]
        return np.concatenate([observation[0].flatten(), observation[1]])  # return the preprocessed observation

ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)
ray.init(log_to_driver=True, object_store_memory=150000000000)
ModelCatalog.register_custom_model("conv_model", LightModel)


def on_episode_start(info):
    for env in info['env'].envs:
        if env.episode_id == -1:
            env.episode_id = info['episode'].episode_id
            info['episode'].global_obs = []
            return

    assert(False)


def on_episode_step(info):
    find_ep = False
    for env in info['env'].envs:
        if env.episode_id == info['episode'].episode_id:
            find_ep = True
            info['episode'].global_obs.append(env.get_global_observation())

    assert(find_ep)


def on_episode_end(info):
    find_ep = False
    for env in info['env'].envs:
        if env.episode_id == info['episode'].episode_id:
            find_ep = True
            info['episode'].global_obs.append(env.get_global_observation())
    assert(find_ep)

def on_episode_end_custom_metric(info):
        episode = info['episode']
        for env in info['env'].envs:
            if env.n_episodes < 6000:
                env.n_episodes += 1
        num_immunized = episode.last_observation_for('agent_0')[-1]
        num_infected = episode.last_observation_for('agent_0')[-2]
        episode.custom_metrics["num_immunized"] = num_immunized
        episode.custom_metrics["num_infected"] = num_infected


def train_func(config, reporter):

    # init the game
    print('Init Env')
    policy_name = config['policy_name'].format(**locals()).replace('.', '')
    
    if config['view_radius'] == 6:
        config['n_agents'] = 8
    elif config['view_radius'] == 3:
        config['n_agents'] = 4
    else:
        raise(NotImplementedError)
    # Specifying observation space and actions space dimensions.
    view_radius = config["view_radius"]*2 + 1
    obs_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=1, shape=(view_radius, view_radius, 4)),
                                  gym.spaces.Space((2,))))
    act_space = (gym.spaces.Discrete(9))

    policy_graphs = {}
    # Dict with the different policies to train
    # policy_graphs['ppo_policy_infection_prob_003'] = (PPOPolicyGraph, obs_space, act_space, {})
    if config['algorithm'] == 'ppo':
        policy_graphs[policy_name] = (PPOPolicyGraph, obs_space, act_space, {})
    elif config['algorithm'] == 'dqn':
        policy_graphs[policy_name] = (DQNPolicyGraph, obs_space, act_space, {})
    # else:
    #     for i in range(config['n_agents']):
    #         policy_graphs[f"ppo_policy_agent_{i}_independent_training_{config['independent_training']}"]\
    #             = (PPOPolicyGraph, obs_space, act_space, {})

    def policy_mapping_fn(agent_id):
        return policy_name  # .replace(str(config['initially_infected']), '1')
        # return "ppo_policy_infection_prob_003"
        # if config['independent_training'] == "common_trainer_common_policy":
        #     return f"ppo_policy_agent_0_vaccine_reward{str(config['vaccine_reward']).replace('.', '')}"
        # else:
        #     return f"ppo_policy_{agent_id}_independent_training_{config['independent_training']}"

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
                  "decreasing_vaccine_reward": config['decreasing_vaccine_reward'],
                  "step_propagation": config['step_propagation']
                  }

    # PPO Config specification
    if config['algorithm'] == 'ppo':
        agent_config = ppo.DEFAULT_CONFIG.copy()
    elif config['algorithm'] == 'dqn':
        agent_config = dqn.DEFAULT_CONFIG.copy()
    
    agent_config['env_config'] = env_config
    # Here we use the default fcnet with modified hidden layers size
    #print('DEFAULT_PREPROCESSOR:', agent_config['preprocessor_pref'])
    agent_config['model'] = {"fcnet_hiddens": config['hidden_sizes'], "custom_preprocessor": "my_prep"}
    #agent_config['model'] = {"custom_model": "conv_model", "custom_preprocessor": "my_prep"}

    agent_config["num_workers"] = 0
    agent_config["num_cpus_per_worker"] = 11
    agent_config["num_gpus"] = 0.5
    agent_config["num_gpus_per_worker"] = 0.5
    agent_config["num_cpus_for_driver"] = 1
    agent_config["num_envs_per_worker"] = 2
    
    if config['algorithm'] == 'ppo':
        agent_config["batch_mode"] = "complete_episodes"
        agent_config["vf_clip_param"] = config['vf_clip_param']
        agent_config["vf_share_layers"] = config['vf_share_layers']
        agent_config["simple_optimizer"] = False
        agent_config["entropy_coeff"] = config["entropy_coeff"]
        agent_config["num_sgd_iter"] = config["num_sgd_iter"]
        agent_config['use_centralized_vf'] = config["use_centralized_vf"]
        agent_config['max_vf_agents'] = 12
        agent_config['sgd_minibatch_size'] = config['sgd_minibatch_size']
        agent_config['clip_param'] = config['clip_param']
        agent_config['gamma'] = config['gamma']
        agent_config['vf_loss_coeff'] = config['vf_loss_coeff']
        agent_config['kl_target'] = config['kl_target']
        agent_config['lr'] = config['learning_rate']

    if config['use_centralized_vf']:
        agent_config['callbacks'] = {
            "on_episode_start": tune.function(on_episode_start),
            "on_episode_step": tune.function(on_episode_step),
            "on_episode_end": tune.function(on_episode_end)}
    agent_config['callbacks'] = {"on_episode_end": tune.function(on_episode_end_custom_metric)}
    agent_config['log_level'] = 'WARN'
    # agent_config["n_step"] = 3

    agent_config['multiagent'] = {"policy_graphs": policy_graphs,
                                "policy_mapping_fn": policy_mapping_fn,
                                "policies_to_train": list(policy_graphs.keys())}


    agent_config['horizon'] = config['horizon']
    def logger_creator(conf):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        logdir = config['folder_name'].format(**locals()).replace('.', '')
        logdir = tempfile.mkdtemp(
            prefix=logdir, dir=config['local_dir'])
        return UnifiedLogger(conf, logdir, None)

    logger = logger_creator
    
    if config['algorithm'] == 'ppo':
        trainer = PPOTrainer(env=GridWorldRLLibEnv, config=agent_config, logger_creator=logger)
    elif config['algorithm'] == 'dqn':
        trainer = DQNTrainer(env=GridWorldRLLibEnv, config=agent_config, logger_creator=logger)
    #ppo_trainer.restore('/home/guillaume/Flatland_Epidemiology/toric_env_tests/3_entropy_coeff_grid_search/ppo_policy_entropy_coeff_0001_mncelqfg/checkpoint_201/checkpoint-201')
    checkpoint_path='/home/guillaume/Flatland_Epidemiology/toric_env_tests/view_range_3_with_step_propagation/ppo_policy_view_range_3_step_propagation_5_infection_prob_015_65coq4y1/checkpoint_201/checkpoint-201'
    state = pickle.load(open(checkpoint_path, "rb"))
    trainer.local_evaluator.restore(state["evaluator"])
    remote_state = ray.put(state["evaluator"])
    for r in trainer.remote_evaluators:
        r.restore.remote(remote_state)

    for i in range(100000 + 2):
        print("== Iteration", i, "==")

        print("-- PPO --")
        print(pretty_print(trainer.train()))

        if i % config['save_every'] == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

        reporter(num_iterations_trained=trainer._iteration)


@gin.configurable
def run_grid_search(name, view_radius, n_agents, hidden_sizes, save_every, map_size, vaccine_reward,
                    vf_clip_param, num_iterations, vf_share_layers, step_reward, final_reward,
                    final_reward_times_healthy, entropy_coeff, local_dir, learning_rate, infection_prob,
                    policy_name, initially_infected, decreasing_vaccine_reward, horizon, use_centralized_vf,
                    num_sgd_iter, sgd_minibatch_size, clip_param, gamma, vf_loss_coeff, kl_target,
                    folder_name, algorithm, step_propagation):

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
                "learning_rate": learning_rate,
                "infection_prob": infection_prob,
                "policy_name": policy_name,
                "initially_infected": initially_infected,
                "decreasing_vaccine_reward": decreasing_vaccine_reward,
                "use_centralized_vf": use_centralized_vf,
                "num_sgd_iter": num_sgd_iter,
                "sgd_minibatch_size": sgd_minibatch_size,
                "clip_param": clip_param,
                "gamma": gamma,
                "vf_loss_coeff": vf_loss_coeff,
                "kl_target": kl_target,
                "folder_name": folder_name,
                "algorithm": algorithm,
                "step_propagation": step_propagation
                },
        resources_per_trial={
            "cpu": 12,
            "gpu": 0.5
        },
        local_dir=local_dir
    )


if __name__ == '__main__':
    gin.external_configurable(tune.grid_search)
    dir = '/home/guillaume/Flatland_Epidemiology/toric_env_tests/view_range_3_large_map_size_with_step_propagation'
    gin.parse_config_file(dir + '/config.gin')
    run_grid_search(local_dir=dir)

