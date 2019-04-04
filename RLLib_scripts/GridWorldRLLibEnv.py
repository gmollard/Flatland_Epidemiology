from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from examples.agent_goal import generate_map

import magent
import numpy as np

import gym
import cv2


@PublicAPI
class GridWorldRLLibEnv(MultiAgentEnv):
    """An environment that hosts multiple independent agents.
    Agents are identified by (string) agent ids. Note that these "agents" here
    are not to be confused with RLlib agents.
    """

    def __init__(self, config):
        super(MultiAgentEnv, self).__init__()
        self.map_size = config["map_size"]
        self.env = magent.GridWorld("agent_goal", map_size=self.map_size)
        self.handles = self.env.get_handles()
        self.render = config['render']
        self.env.set_render_dir("build/render")

        # self.handles = config["handles"]
        self.agent_generator = config["agent_generator"]
        self.num_static_blocks = 1
        if "num_static_blocks" in config.keys():
            self.num_static_blocks = config["num_static_blocks"]
        self.action_space = gym.spaces.Discrete(9)
        self.action_space = None
        # self.observation_space = gym.spaces.Tuple((gym.spaces.Space((31,31,6)), gym.spaces.Space((21,))))
        self.observation_space = gym.spaces.Space((42,42,6))
        self.observation_space = None



    @PublicAPI
    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        self.env.reset()
        generate_map(self.env, self.map_size, self.handles, self.agent_generator)
        observations = self.env.get_observation(self.handles[1])
        self.agents = [f'agent_{i}' for i in range(self.env.get_num(self.handles[1]))]
        obs = {}
        for i, agent_name in enumerate(self.agents):
            obs[agent_name] = [observations[0][i], observations[1][i]]#, observations[1][i]]

        self.total_reward = 0
        self.num_infected = 1
        return obs

    @PublicAPI
    def step(self, action_dict):
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        self.env.set_action(self.handles[1], np.array([action_dict[agent_name]\
                                                       for agent_name in self.agents]).astype(np.int32))
        # # print(action_dict)
        # self.env.set_action(self.handles[1], np.array([action_dict]).astype(np.int32))
        done = self.env.step()
        rew = self.env.get_reward(self.handles[1])
        # rew *= len(rew)
        observations = self.env.get_observation(self.handles[1])
        obs = {}
        rewards = {}
        dones = {}
        dones['__all__'] = self.env.epidemy_contained() or done
        if self.env.get_num_infected(self.handles[0])[0] > self.num_infected:
            # rew -= self.env.get_num_infected(self.handles[0])[0] - self.num_infected # self.env.get_num_infected(self.handles[0])
            self.num_infected = self.env.get_num_infected(self.handles[0])[0]

        self.total_reward += sum(rew)
        if dones['__all__']:
            rew += 10*(self.env.get_num(self.handles[0]) -
                        (self.env.get_num_immunized(self.handles[0]) + self.env.get_num_infected(self.handles[0]))) / len(rew)

            if (self.env.get_num(self.handles[0]) -
                        (self.env.get_num_immunized(self.handles[0]) + self.env.get_num_infected(self.handles[0]))) > 0:
                print('DONE !!!!!')
                # for i in range(6):
                #     for j in range(4):
                #         cv2.imwrite(f'obs_{j}_{i}.png', observations[0][0][:,:,i]*255.0)

            print(self.num_infected)
            print(self.total_reward + 10*(self.env.get_num(self.handles[0]) -
                        (self.env.get_num_immunized(self.handles[0]) + self.env.get_num_infected(self.handles[0]))))

        infos = {}
        for i, agent_name in enumerate(self.agents):
            obs[agent_name] = [observations[0][i], observations[1][i]]
            rewards[agent_name] = rew[i] / self.num_static_blocks
            dones[agent_name] = False
            infos[agent_name] = None
        # clear dead agents
        self.env.clear_dead()

        if self.render:
            self.env.render()
        return obs, rewards, dones, {}

