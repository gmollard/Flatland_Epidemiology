


if __name__ == "__main__":
    ray.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--n_round", type=int, default=50000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # init the game
    print('Init Env')
    map_size = 100

    # Specifying observation space and actions space dimensions.
    # TODO: take this as parameter and send it to the corresponding magent/builtin/config file.
    obs_space = gym.spaces.Tuple((gym.spaces.Box(low=0, high=1, shape=(31, 31, 6)),
                                                gym.spaces.Box(low=0, high=1, shape=(21,) )))
    act_space = (gym.spaces.Discrete(9))

    # Dict with the different policies to train
    policy_graphs = {
        "ppo_policy_agent_0": (PPOPolicyGraph, obs_space, act_space, {}),
        # "ppo_policy_agent_1_vaccine_reward_01_vf_clip_param_10": (PPOPolicyGraph, obs_space, act_space, {}),
        # "ppo_policy_agent_2_vaccine_reward_01_vf_clip_param_10": (PPOPolicyGraph, obs_space, act_space, {}),
        # "ppo_policy_agent_3_vaccine_reward_01_vf_clip_param_10": (PPOPolicyGraph, obs_space, act_space, {}),
        # "ppo_policy_agent_3": (PPOPolicyGraph, obs_space, act_space, {}),
    }

    # Function mapping agent id to the corrresponding policy
    def policy_mapping_fn(agent_id):
        return f"ppo_policy_agent_0"
        # if agent_id == "agent_0":
        #     return "ppo_policy_agent_0"
        # return "ppo_policy_agent_1"

    # Environment configuration
    env_config = {"map_size": args.map_size,
            "agent_generator": args.agent_generator,
            "render": args.render,
            "num_static_blocks": 1,
            "n_agents": 4,
            "vaccine_reward": 0.1
    }

    register_env("gridworld", lambda _: GridWorldRLLibEnv(env_config))


    # PPO Config specification
    config = ppo.DEFAULT_CONFIG.copy()
    # config['model'] = {"use_lstm": True}
    # config['model'] = {"custom_model": "light_model"}
    config['model'] = {"fcnet_hiddens": [64, 64]}  # Here we u0e the default fcnet with modified hidden layers size

    config["num_workers"] = 0
    config["num_cpus_per_worker"] = 15
    config["num_gpus"] = 0.5
    config["num_gpus_per_worker"] = 0.5
    config["num_cpus_for_driver"] = 8
    config["num_envs_per_worker"] = 1

    # Config for rendering (Only one environment in parallel or there is a bug with de video.txt file.
    # config["num_workers"] = 0
    # config["num_cpus_per_worker"] = 15
    # config["num_gpus"] = 1
    # config["num_cpus_for_driver"] = 1
    # config["num_envs_per_worker"] = 1

    config['multiagent'] = {"policy_graphs": policy_graphs,
                            "policy_mapping_fn": policy_mapping_fn,
                            "policies_to_train": list(policy_graphs.keys())}  # , "ppo_policy_agent_1", "ppo_policy_agent_2",
    # "ppo_policy_agent_3"]

    # def logger_creator(conf):
    #     """Creates a Unified logger with a default logdir prefix
    #     containing the agent name and the env id
    #     """
    #     logdir = f"ppo_4_agents_4_policies"
    #     logdir = tempfile.mkdtemp(
    #         prefix=logdir, dir="/mount/SDC/ray_results_long_run")
    #     return UnifiedLogger(conf, logdir, None)

    # logger = logger_creator

    ppo_trainer = PPOAgent(env="gridworld", config=config)#, logger_creator=logger)

    # To reload policies from a checkpoint
    ppo_trainer.restore('/mount/SDC/ray_results_ppo_rewaard_benchamark/ppo_vaccine_reward_01_vf_clip_param_1049dx2247/checkpoint_2801/checkpoint-2801')


    for i in range(args.n_round + 2):
        print("== Iteration", i, "==")

        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))

        # if i % args.save_every == 0:
        #     checkpoint = ppo_trainer.save()
        #     print("checkpoint saved at", checkpoint)