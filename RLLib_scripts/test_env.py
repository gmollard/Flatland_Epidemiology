from RLLib_scripts.GridWorldRLLibEnv import GridWorldRLLibEnv
import numpy as np

# Environment configuration
env_config = {"map_size": 19,
              "agent_generator": 'large_toric_env',
              "render": True,
              "num_static_blocks": 1,
              "n_agents": None,
              "vaccine_reward": 0.1,
              "view_radius": 6,
              "step_reward": 0.0,
              "final_reward": 10,
              "final_reward_times_healthy": True,
              "bad_vaccine_penalty": -0.1,
              "collide_penalty": -0.1,
              "initially_infected": 1,
              "infection_prob": 0.2,
              "decreasing_vaccine_reward": False
              }

env = GridWorldRLLibEnv(env_config)


def pprint(obs_array):
    out = np.zeros_like(obs_array).astype(str)
    for index in np.argwhere(obs_array > 0.5):
        out[index[0], index[1]] = 'O'

    for index in np.argwhere(obs_array < 0.5):
        out[index[0], index[1]] = ' '
    # print(np.argwhere(obs_array > 0.5))
    # out[np.argwhere(obs_array > 0.5)] = 'O'
    # out[np.argwhere(obs_array < 0.5)] = '~'
    print(out)


obs = env.reset()
action_dict = dict()
for k in obs.keys():
    action_dict[k] = 5

for i in range(5):
    obs, _, _, _ = env.step(action_dict)
    print('healthy agents')
    pprint(obs['agent_0'][0][:, :, 1])
    print('infected agents')
    pprint(obs['agent_0'][0][:, :, 2])
    print()

# print("reward:", env.step({"agent_0": 5})[1])
# print("reward:", env.step({"agent_0": 4})[1])
# print("reward:", env.step({"agent_0": 7})[1])
# print("reward:", env.step({"agent_0": 6})[1])
# print("reward:", env.step({"agent_0": 5})[1])
# print("reward:", env.step({"agent_0": 5})[1])
# print("reward:", env.step({"agent_0": 5})[1])
# print("reward:", env.step({"agent_0": 8})[1])
# for i in range(13):
#     print("reward:", env.step({"agent_0": 3})[1])
# env.step({"agent_0": 4})
# env.step({"agent_0": 6})
# env.step({"agent_0": 7})
# env.step({"agent_0": 3})
# for _ in range(5):
#     env.step({"agent_0": 4})
# env.step({"agent_0": 1})
# env.step({"agent_0": 1})
# env.step({"agent_0": 4})
# env.step({"agent_0": 4})
# env.step({"agent_0": 1})
# print("reward:", env.step({"agent_0": 8})[1])
# print(env.step({"agent_0": 8})[0][ 'agent_0'][0][:,:,1])

# obs = [obs['agent_0'][0][:, :, j] for j in range(4)]
#
# print(obs[1])

# for j in range(4):
#     print(obs[j])
