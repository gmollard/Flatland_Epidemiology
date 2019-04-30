from RLLib_scripts.GridWorldRLLibEnv import GridWorldRLLibEnv

# Environment configuration
env_config = {"map_size": 19,
              "agent_generator": 'toric_env',
              "render": True,
              "num_static_blocks": 1,
              "n_agents": 1,
              "vaccine_reward": 0.1,
              "view_radius": 5
              }

env = GridWorldRLLibEnv(env_config)

obs = env.reset()

env.step({"agent_0": 5})
env.step({"agent_0": 8})
for i in range(13):
    env.step({"agent_0": 3})
env.step({"agent_0": 4})
env.step({"agent_0": 6})
env.step({"agent_0": 7})
env.step({"agent_0": 3})
for _ in range(5):
    env.step({"agent_0": 4})
env.step({"agent_0": 1})
env.step({"agent_0": 1})
env.step({"agent_0": 4})
env.step({"agent_0": 4})
env.step({"agent_0": 1})
print(env.step({"agent_0": 8})[0]['agent_0'][0][:,:,1])

# obs = [obs['agent_0'][0][:, :, j] for j in range(4)]
#
# print(obs[1])

# for j in range(4):
#     print(obs[j])
