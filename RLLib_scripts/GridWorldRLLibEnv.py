from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from env_generator.agent_goal import generate_map

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
        self.step_reward = config["step_reward"]
        self.view_radius = config["view_radius"]
        self.vaccine_reward = config['vaccine_reward']
        self.final_reward = config['final_reward']
        self.bad_vaccine_penalty = config['bad_vaccine_penalty']
        self.collide_penalty = config['collide_penalty']
        self.final_reward_times_healthy = config["final_reward_times_healthy"]
        self.infection_prob = config['infection_prob']
        self.initially_infected = config['initially_infected']
        self.env = magent.GridWorld("agent_goal", map_size=self.map_size,
                                    vaccine_reward=self.vaccine_reward, view_radius=self.view_radius,
                                    step_reward=self.step_reward, infection_prob=self.infection_prob)
        self.handles = self.env.get_handles()
        self.render = config['render']
        self.n_agents = None
        if 'n_agents' in config.keys():
            self.n_agents = config['n_agents']
        if config["render"]:
            self.env.set_render_dir("build/render")

        # self.handles = config["handles"]
        self.agent_generator = config["agent_generator"]
        # self.observation_space = gym.spaces.Tuple((gym.spaces.Space((31,31,6)), gym.spaces.Space((21,))))
        self.observation_mode = "dist_map"#config["view_mode"]
        self.decreasing_vaccine_reward = config["decreasing_vaccine_reward"]

        self.n_reset = 0




    @PublicAPI
    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        self.env.reset()
        generate_map(self.env, self.map_size, self.handles, self.agent_generator, self.n_agents,
                     n_infected_init=self.initially_infected)
        observations = self.env.get_observation(self.handles[1], self.observation_mode)
        # assert(self.n_agents == self.env.get_num(self.handles[1]))
        self.agents = [f'agent_{i}' for i in range(self.env.get_num(self.handles[1]))]
        obs = {}
        for i, agent_name in enumerate(self.agents):
            obs[agent_name] = [observations[0][i], observations[1][i]]#, observations[1][i]]

        if self.render:
            self.env.render()

        self.total_reward = 0
        self.num_infected = 1
        self.step_count = 0
        if self.n_reset < 10000:
            self.n_reset += 1

        # self.count_step = 0
        for j in range(4):
            cv2.imwrite(f'obs_{j}.png', obs['agent_0'][0][:, :, j] * 255.0)

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
        # self.env.set_action(self.handles[1], np.array([3 \
        #                                                for agent_name in self.agents]).astype(np.int32))
        # # print(action_dict)
        # self.env.set_action(self.handles[1], np.array([action_dict]).astype(np.int32))
        done = self.env.step()
        rew = self.env.get_reward(self.handles[1])
        # rew *= len(rew)
        observations = self.env.get_observation(self.handles[1], self.observation_mode)
        obs = {}
        rewards = {}
        dones = {}
        dones['__all__'] = self.env.epidemy_contained() or done
        if self.env.get_num_infected(self.handles[0])[0] > self.num_infected:
            # rew -= self.env.get_num_infected(self.handles[0])[0] - self.num_infected # self.env.get_num_infected(self.handles[0])
            self.num_infected = self.env.get_num_infected(self.handles[0])[0]

        self.total_reward += sum(rew)

        for i in range(len(rew)):
            if rew[i] > self.step_reward:
                rew[i] -= 2*self.vaccine_reward*(self.n_reset / 10000)

        if dones['__all__']:
            if self.final_reward_times_healthy:
                rew += self.final_reward*(self.env.get_num(self.handles[0]) -
                        (self.env.get_num_immunized(self.handles[0]) + self.env.get_num_infected(self.handles[0])))
            else:
                rew += self.final_reward

            if (self.env.get_num(self.handles[0]) -
                        (self.env.get_num_immunized(self.handles[0]) + self.env.get_num_infected(self.handles[0]))) > 0:
                print('DONE !!!!!')
                # for i in range(6):
                #     for j in range(4):
                #         cv2.imwrite(f'obs_{j}_{i}.png', observations[0][0][:,:,i]*255.0)

            print('Infected:', self.num_infected)
            print('Immunized:', self.env.get_num_immunized(self.handles[0]))
            #print(self.total_reward + 10*(self.env.get_num(self.handles[0]) -
            #            (self.env.get_num_immunized(self.handles[0]) + self.env.get_num_infected(self.handles[0]))))

        infos = {}
        for i, agent_name in enumerate(self.agents):
            obs[agent_name] = [observations[0][i], observations[1][i]]
            obs[agent_name][1][-1] = self.total_reward
            rewards[agent_name] = rew[i]
            dones[agent_name] = False
            infos[agent_name] = None
        # clear dead agents
        self.env.clear_dead()

        if self.render:
            self.env.render()



        # if self.count_step == 50:
        #     for j in range(7):
        #         cv2.imwrite(f'obs_{j}.png', obs['agent_1'][0][:, :, j] * 255.0)
        #     print(ds)
        #
        # self.count_step += 1

        return obs, rewards, dones, {}


