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

import ray
from ray.rllib.agents.dqn.dqn import DQNAgent
from ray.rllib.agents.dqn.dqn_policy_graph import DQNPolicyGraph
from ray.rllib.agents.ppo.ppo import PPOAgent
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.tests.test_multi_agent_env import MultiCartpole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from magent import GridWorldRLLibEnv
import magent

parser = argparse.ArgumentParser()
parser.add_argument("--num-iters", type=int, default=20)


if __name__ == "__main__":
    ray.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--n_round", type=int, default=100001)
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

    env = magent.GridWorld("agent_goal", map_size=args.map_size)
    print('Env initialized')
    env.set_render_dir("build/render")

    # two groups of animal
    deer_handle, tiger_handle = env.get_handles()
    handles = [deer_handle, tiger_handle]
    obs_space = env.get_view_space(tiger_handle)
    act_space = env.get_action_space(tiger_handle)


    def env_creator(env_config):
        return GridWorldRLLibEnv(None, args.map_size, handles, args.agent_generator)

    # Simple environment with 4 independent cartpole entities
    register_env("gridworld", lambda _: env_creator(None))

    # You can also have multiple policy graphs per trainer, but here we just
    # show one each for PPO and DQN.
    policy_graphs = {
        # "ppo_policy": (PPOPolicyGraph, obs_space, act_space, {}),
        "dqn_policy": (DQNPolicyGraph, obs_space, act_space, {}),
    }

    # def policy_mapping_fn(agent_id):
    #     if agent_id % 2 == 0:
    #         return "ppo_policy"
    #     else:
    #         return "dqn_policy"

    dqn_trainer = DQNAgent(
        env="gridworld",
        config={
            "multiagent": {
                "policy_graphs": policy_graphs,
                "policy_mapping_fn": 'dqn_policy',
                "policies_to_train": ["dqn_policy"],
            },
            "gamma": 0.95,
            "n_step": 3,
        })

    # You should see both the printed X and Y approach 200 as this trains:
    # info:
    #   policy_reward_mean:
    #     dqn_policy: X
    #     ppo_policy: Y
    for i in range(args.num_iters):
        print("== Iteration", i, "==")

        # improve the DQN policy
        print("-- DQN --")
        print(pretty_print(dqn_trainer.train()))

        # improve the PPO policy
        # print("-- PPO --")
        # print(pretty_print(ppo_trainer.train()))

        # swap weights to synchronize
        dqn_trainer.set_weights(dqn_trainer.get_weights(["ppo_policy"]))
