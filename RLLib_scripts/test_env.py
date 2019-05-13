from RLLib_scripts.GridWorldRLLibEnv import GridWorldRLLibEnv

# Environment configuration
env_config = {"map_size": 19,
              "agent_generator": 'toric_env',
              "render": True,
              "num_static_blocks": 1,
              "n_agents": 1,
              "vaccine_reward": 0.1,
              "view_radius": 5,
              "step_reward": 0.0,
              "final_reward": 10,
              "final_reward_times_healthy": True,
              "bad_vaccine_penalty": -0.1,
              "collide_penalty": -0.1
              }

env = GridWorldRLLibEnv(env_config)

obs = env.reset()

print("reward:", env.step({"agent_0": 5})[1])
print("reward:", env.step({"agent_0": 4})[1])
print("reward:", env.step({"agent_0": 7})[1])
print("reward:", env.step({"agent_0": 6})[1])
print("reward:", env.step({"agent_0": 5})[1])
print("reward:", env.step({"agent_0": 5})[1])
print("reward:", env.step({"agent_0": 5})[1])
print("reward:", env.step({"agent_0": 8})[1])
for i in range(13):
    print("reward:", env.step({"agent_0": 3})[1])
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
print("reward:", env.step({"agent_0": 8})[1])
# print(env.step({"agent_0": 8})[0]['agent_0'][0][:,:,1])

# obs = [obs['agent_0'][0][:, :, j] for j in range(4)]
#
# print(obs[1])

# for j in range(4):
#     print(obs[j])