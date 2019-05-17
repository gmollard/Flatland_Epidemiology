"""
Double attack, tigers get reward when they attack a same deer
"""

import argparse
import time
import logging as log

import numpy as np

import magent
from magent.builtin.rule_model import RandomActor
import cv2
import os

reward_array = []
def generate_map(env, map_size, handles, agent_generator, n_agents=None, infection_range=None, n_infected_init=1):
    # env.add_walls(method="random", n=map_size*map_size*0.04)
    if agent_generator == 'random_clusters':
        n_agent_per_cluster = 81
        n_cluster_tiger = int(map_size*map_size*0.01 /n_agent_per_cluster)
        n_cluster_deer = int(map_size*map_size*0.03 /n_agent_per_cluster)

        for _ in range(n_cluster_tiger):
            x = np.random.randint(np.sqrt(n_agent_per_cluster), map_size - np.sqrt(n_agent_per_cluster))
            y = np.random.randint(np.sqrt(n_agent_per_cluster), map_size - np.sqrt(n_agent_per_cluster))
            tiger_pos = []
            for i in range(int(-np.sqrt(n_agent_per_cluster)/2)-1, int(np.sqrt(n_agent_per_cluster)/2)):
                for j in range(int(-np.sqrt(n_agent_per_cluster)/2)-1, int(np.sqrt(n_agent_per_cluster)/2)):
                    tiger_pos.append((x+i, y+j))

            env.add_agents(handles[1], method="custom", pos=tiger_pos)

        for _ in range(n_cluster_deer):
            x = np.random.randint(np.sqrt(n_agent_per_cluster), map_size - np.sqrt(n_agent_per_cluster))
            y = np.random.randint(np.sqrt(n_agent_per_cluster), map_size - np.sqrt(n_agent_per_cluster))
            deer_pos = []
            for i in range(int(-np.sqrt(n_agent_per_cluster) / 2) - 1, int(np.sqrt(n_agent_per_cluster) / 2)):
                for j in range(int(-np.sqrt(n_agent_per_cluster) / 2) - 1, int(np.sqrt(n_agent_per_cluster) / 2)):
                    deer_pos.append((x + i, y + j))

            env.add_agents(handles[0], method="custom", pos=deer_pos)

    elif agent_generator == 'random_static_clusters':
        x_coords = np.arange(10, map_size - 10, 20)
        y_coords = np.arange(10, map_size - 10, 20)
        deer_pos = []
        for x in x_coords:
            for y in y_coords:
                for i in range(x + 1, x + 17, 2):
                    for j in range(y + 1, y + 17, 2):
                        deer_pos.append((i, j))

        # infected_ids = [np.random.randint(0, 64), 64 + np.random.randint(0, 64),
        #                 64 * 2 + np.random.randint(0, 64), 64 * 3 + np.random.randint(0, 64)]
        infected_ids = []
        for i in range(len(x_coords)*len(y_coords)):
            infected_ids.append(np.random.randint(0,64) + i*64)

        env.add_agents(handles[0], method="custom_infection", pos=deer_pos, infected=infected_ids)

        tiger_pos = []
        for x in x_coords:
            for y in y_coords:
               # tiger_pos.append((x + 8, y+17))
               # tiger_pos.append((x + 8, y - 2))
               tiger_pos.append((x - 2, y+8))
               tiger_pos.append((x +17, y+8))
                # dir = np.random.uniform(-1, 1, 2)
                # dir = dir / np.linalg.norm(dir)
                # tiger_pos.append((x + 6 + dir[0] * 6, y + 6 + dir[1] * 6))
        env.add_agents(handles[1], method="custom", pos=tiger_pos)

    elif agent_generator == 'random_static_clusters_1_to_4_agents':
        x_coords = np.arange(10, map_size - 10, 20)
        y_coords = np.arange(10, map_size - 10, 20)
        deer_pos = []
        for x in x_coords:
            for y in y_coords:
                for i in range(x + 1, x + 17, 2):
                    for j in range(y + 1, y + 17, 2):
                        deer_pos.append((i, j))

        # infected_ids = [np.random.randint(0, 64), 64 + np.random.randint(0, 64),
        #                 64 * 2 + np.random.randint(0, 64), 64 * 3 + np.random.randint(0, 64)]
        infected_ids = []
        for i in range(len(x_coords)*len(y_coords)):
            infected_ids.append(np.random.randint(0,64) + i*64)

        env.add_agents(handles[0], method="custom_infection", pos=deer_pos, infected=infected_ids)

        tiger_pos = []
        for x in x_coords:
            for y in y_coords:
               tiger_pos.append((x + 8, y+17))
               tiger_pos.append((x + 8, y - 2))
               tiger_pos.append((x - 2, y+8))
               tiger_pos.append((x +17, y+8))
                # dir = np.random.uniform(-1, 1, 2)
                # dir = dir / np.linalg.norm(dir)
                # tiger_pos.append((x + 6 + dir[0] * 6, y + 6 + dir[1] * 6))
        env.add_agents(handles[1], method="custom", pos=tiger_pos[:n_agents])

    elif agent_generator == 'scale_map_size_4_agents':

        deer_pos = []

        for i in range(10 + 1, map_size - 10, 2):
            for j in range(10 + 1, map_size - 10, 2):
                deer_pos.append((i, j))

        # infected_ids = [np.random.randint(0, 64), 64 + np.random.randint(0, 64),
        #                 64 * 2 + np.random.randint(0, 64), 64 * 3 + np.random.randint(0, 64)]
        infected_ids = [np.random.randint(0, len(deer_pos))]

        env.add_agents(handles[0], method="custom_infection", pos=deer_pos, infected=infected_ids)

        tiger_pos = []

        tiger_pos.append((map_size / 2, map_size - 10))
        tiger_pos.append((map_size / 2, 8))
        tiger_pos.append((8, map_size / 2))
        tiger_pos.append((map_size - 10, map_size / 2))
        env.add_agents(handles[1], method="custom", pos=tiger_pos)

    elif agent_generator == 'random_static_clusters_single_agent':
        x_coords = np.arange(10, map_size - 10, 20)
        y_coords = np.arange(10, map_size - 10, 20)
        deer_pos = []
        for x in x_coords:
            for y in y_coords:
                for i in range(x + 1, x + 17, 2):
                    for j in range(y + 1, y + 17, 2):
                        deer_pos.append((i, j))

        # infected_ids = [np.random.randint(0, 64), 64 + np.random.randint(0, 64),
        #                 64 * 2 + np.random.randint(0, 64), 64 * 3 + np.random.randint(0, 64)]
        infected_ids = []
        for i in range(len(x_coords) * len(y_coords)):
            infected_ids.append(np.random.randint(0, 64) + i * 64)

        env.add_agents(handles[0], method="custom_infection", pos=deer_pos, infected=infected_ids)

        tiger_pos = []
        x = x_coords[0]
        y = y_coords[0]

        tiger_pos.append((x + 8, y + 17))
        tiger_pos.append((x + 8, y - 2))
        tiger_pos.append((x - 2, y + 8))
        tiger_pos.append((x + 17, y + 8))

        env.add_agents(handles[1], method="custom", pos=[tiger_pos[np.random.randint(0, 4)]])

    elif agent_generator == 'random_static_clusters_two_agents':
        x_coords = np.arange(10, map_size - 10, 20)
        y_coords = np.arange(10, map_size - 10, 20)
        deer_pos = []
        for x in x_coords:
            for y in y_coords:
                for i in range(x + 1, x + 17, 2):
                    for j in range(y + 1, y + 17, 2):
                        deer_pos.append((i, j))

        # infected_ids = [np.random.randint(0, 64), 64 + np.random.randint(0, 64),
        #                 64 * 2 + np.random.randint(0, 64), 64 * 3 + np.random.randint(0, 64)]
        infected_ids = []
        for i in range(len(x_coords) * len(y_coords)):
            infected_ids.append(np.random.randint(0, 64) + i * 64)

        env.add_agents(handles[0], method="custom_infection", pos=deer_pos, infected=infected_ids)

        tiger_pos = []
        x = x_coords[0]
        y = y_coords[0]

        # tiger_pos.append((x + 8, y + 17))
        tiger_pos.append((x + 8, y - 2))
        # tiger_pos.append((x - 2, y + 8))
        tiger_pos.append((x + 17, y + 8))

        env.add_agents(handles[1], method="custom", pos=tiger_pos)



    elif agent_generator == 'two_clusters':
        x = y = map_size / 2
        n_deers = 81
        deer_pos = []


        for i in range(int(-np.sqrt(n_deers) / 2) - 1, int(np.sqrt(n_deers) / 2)):
            for j in range(int(-np.sqrt(n_deers) / 2) - 1, int(np.sqrt(n_deers) / 2)):
                deer_pos.append((x + i, y + j))

        env.add_agents(handles[0], method="custom", pos=deer_pos)

        n_tigers = 4
        tiger_view_r = int(env.get_view_space(handles[1])[0] / 3)
        dir = np.random.uniform(-1, 1, 2)
        dir = dir / np.linalg.norm(dir)
        dir = (-1, 0)
        x = x + dir[0] * tiger_view_r
        y = y + dir[1] * tiger_view_r
        tiger_pos = []
        for i in range(int(-np.sqrt(n_tigers) / 2) - 1, int(np.sqrt(n_tigers) / 2)):
            for j in range(int(-np.sqrt(n_tigers) / 2) - 1, int(np.sqrt(n_tigers) / 2)):
                tiger_pos.append((x + i, y + j))

        env.add_agents(handles[1], method="custom", pos=tiger_pos)

    elif agent_generator == 'static_cluster_spaced':
        deer_pos = []
        for i in range(3, 11, 2):
            for j in range(3, 11, 2):
                deer_pos.append((i, j))

        env.add_agents(handles[0], method="custom", pos=deer_pos)

        tiger_pos = [(4, 1), (6, 1)]
        env.add_agents(handles[1], method="custom", pos=tiger_pos)


    elif agent_generator == 'static_cluster_spaced_start_br_corner':
        deer_pos = []
        for i in range(3, 11, 2):
            for j in range(3, 11, 2):
                deer_pos.append((i, j))

        env.add_agents(handles[0], method="custom", pos=deer_pos)

        tiger_pos = [(4, 1), (6, 1)]
        env.add_agents(handles[1], method="custom", pos=tiger_pos)

    elif agent_generator == "randomized_init":
        import time
        t0 = time.time()
        n_health_agents = np.random.randint(n_agents[0], n_agents[1])
        n_pop_agents = int(map_size ** 2 / 8)

        available_pos = []
        occupied_pos = []

        def add_pos(init_pos=None):
            if init_pos is not None:
                print(init_pos)
                pos = init_pos
            else:
                while True:
                    pos = available_pos[np.random.randint(0, len(available_pos))]
                    if pos not in occupied_pos:
                        break

            occupied_pos.append(pos)
            if pos[0]-2 > 1:
                available_pos.append((pos[0]-2, pos[1]))
            if pos[0] + 2 < map_size-1:
                available_pos.append((pos[0]+2, pos[1]))
            if pos[1] - 2 > 1:
                available_pos.append((pos[0], pos[1]-2))
            if pos[1] + 2 < map_size-1:
                available_pos.append((pos[0], pos[1]+2))

        # We initialize the first agent in a sub square of size map_size / 2
        # to ensure the population won't lie too much next to the environment border.
        add_pos(tuple(np.random.randint(int(map_size / 4), int(3*map_size / 4), 2)))
        for i in range(n_pop_agents - 1):
            add_pos()

        infected_id = np.random.randint(0, n_pop_agents)
        print(infected_id)
        env.add_agents(handles[0], method="custom_infection", pos=occupied_pos, infected=[infected_id])
                       # infected=[np.random.randint(0, n_pop_agents)])
        print('population_initialized')
        env.add_agents(handles[1], method="random", n=n_health_agents)

        print('Init time:', time.time()-t0)

    elif agent_generator == 'toric_env':
        population_pos = []
        for i in range(0, map_size-1, 2):
            for j in range(0, map_size-1, 2):
                population_pos.append((i, j))

        health_officials_pos = [(map_size / 4, map_size / 2), (map_size / 2, map_size / 4),
                                (3 * map_size / 4, map_size / 2), (map_size / 2, 3 * map_size / 4)]
        if (map_size / 4) % 2 == 0 and (map_size / 2) % 2 == 0:
            health_officials_pos = [(map_size / 4 + 1, map_size / 2), (map_size / 2, map_size / 4 + 1),
                                    (3 * map_size / 4 + 1, map_size / 2), (map_size / 2, 3 * map_size / 4 + 1)]

        infected_id = np.random.randint(0, len(population_pos))
        env.add_agents(handles[0], method="custom_infection", pos=population_pos, infected=[infected_id])

        env.add_agents(handles[1], method="custom", pos=health_officials_pos)
    elif agent_generator == 'large_toric_env':
        population_pos = []
        for i in range(0, map_size-1, 2):
            for j in range(0, map_size-1, 2):
                population_pos.append((i, j))

        health_officials_pos = []
        for i in range(9, map_size, 16):
            for j in range(5, map_size - 1, 8):
                health_officials_pos.append((i, j))
	
        for i in range(5, map_size, 8):
            for j in range(9, map_size - 1, 16):
            	health_officials_pos.append((i, j))
	
        #print('POP POS', population_pos)
        #print('HEALTH POS', health_officials_pos)
        infected_ids = np.random.choice(range(len(population_pos)), n_infected_init, replace=False)

        env.add_agents(handles[0], method="custom_infection", pos=population_pos, infected=infected_ids)

        env.add_agents(handles[1], method="custom", pos=health_officials_pos)


    else:
        env.add_agents(handles[0], method="random", n=map_size*map_size*0.1)
        env.add_agents(handles[1], method="random", n=map_size*map_size*0.05)


def play_a_round(env, map_size, handles, models, print_every, agent_generator,
                 train_id=1, step_batch_size=None, render=False, eps=None):

    global reward_array
    env.reset()
    generate_map(env, map_size, handles, agent_generator)

    step_ct = 0
    total_reward = 0
    done = False
    total_loss = value = 0

    n = len(handles)
    obs  = [[] for _ in range(n)]
    ids  = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [0 for _ in range(n)]
    sample_buffer = magent.utility.EpisodesBuffer(10000)
    n_transition = 0

    print("===== sample =====")
    print("eps %s" % eps)
    start_time = time.time()

    n_step = 0
    ended = False
    while not done and not ended:
        # take actions for every model
        for i in range(n):
            if i == 0:
                temp_num = env.get_num(handles[i])
                obs[i] = (np.empty(temp_num), np.empty(temp_num))
            else:
                obs[i] = env.get_observation(handles[i])

            ids[i] = env.get_agent_id(handles[i])
            acts[i] = models[i].infer_action(obs[i], ids[i], policy='e_greedy', eps=eps)
            env.set_action(handles[i], acts[i])

            # if i == 1 and n_step > 0:
            #     print(obs[1][1][0])
            #     for j in range(6):
            #         cv2.imwrite(f'obs_{n_step}_{j}.png', 255*obs[i][0][0][:,:,j])

        # simulate one step
        n_step += 1
        done = env.step()

        # if n_step == 100:
        #     ended = True
        #     done = True

        if n_step == 1:
            old_num_infected = 4
        else:
            old_num_infected = num_infected

        num_infected = env.get_num_infected(handles[0])


        reward = 0

        rewards = env.get_reward(handles[train_id])
        if env.epidemy_contained():
            done = True
            # rewards *= 0
            # rewards += 10*(env.get_num(handles[0]) -
            #             (env.get_num_immunized(handles[0]) + env.get_num_infected(handles[0])))
        rewards -= env.get_num_infected(handles[0])
        # print(sum(env.get_reward(handles[train_id])) / env.get_num(handles[train_id]))
        # rewards += sum(env.get_reward(handles[0])) / env.get_num(handles[train_id])
            # print(env.get_reward(handles[1]))
        # assert(n_step < 10)

        alives = env.get_alive(handles[train_id])

        # rewards -= (num_infected - old_num_infected) / env.get_num(handles[train_id])

        assert(env.get_num_infected(handles[0]) + env.get_num_immunized(handles[0]) <=  env.get_num(handles[0]),
            "Some vaccined agents are also infected !!!!!!!!!!!!!!")

        # if done:
        #     rewards += (env.get_num(handles[0]) - env.get_num_infected(handles[0]))*10 / env.get_num(handles[train_id])



        reward = sum(rewards)
        total_reward += reward
        if done and train_id != -1:
            if not reward_array:
                if os.path.exists('reward_array.npy'):
                    reward_array = list(np.load('reward_array.npy'))
            reward_array.append(total_reward)
            if (n_step % 2) == 0:
                np.save('reward_array.npy', np.array(reward_array))

        if train_id != -1:
            sample_buffer.record_step(ids[train_id], obs[train_id], acts[train_id], rewards, alives)


        # for i in range(n):
        #     rewards = env.get_reward(handles[i])
        #     if train_id != -1 and i == train_id:
        #         alives = env.get_alive(handles[train_id])
        #         sample_buffer.record_step(ids[train_id], obs[train_id], acts[train_id], rewards, alives)
        #     reward = sum(rewards)
        #     total_reward += reward


        # render
        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        # stats info
        for i in range(n):
            nums[i] = env.get_num(handles[i])
        n_transition += nums[train_id]


        if step_ct % print_every == 0:
            print("step %3d,  deer: %5d,  tiger: %5d,  train_id: %d,  reward: %.2f,  total_reward: %.2f " %
                  (step_ct, nums[0], nums[1], train_id, reward, total_reward))
        step_ct += 1
        if step_ct > 1000:
            break

        if step_batch_size and n_transition > step_batch_size and train_id != -1:
            total_loss, value = models[train_id].train(sample_buffer, 500)
            sample_buffer.reset()
            n_transition = 0



    sample_time = time.time() - start_time
    print("steps: %d, total time: %.2f, step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    # train
    if train_id != -1:
        print("===== train =====")
        start_time = time.time()
        total_loss, value = models[train_id].train(sample_buffer)
        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    return total_loss, total_reward, value


if __name__ == "__main__":
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
        args.map_size = 40#55

    env = magent.GridWorld("agent_goal", map_size=args.map_size)
    print('Env initialized')
    env.set_render_dir("build/render")

    # two groups of animal
    deer_handle, tiger_handle = env.get_handles()

    # init two models
    models = [
        RandomActor(env, deer_handle, tiger_handle),
    ]

    batch_size = 128
    unroll     = 8

    if args.alg == 'dqn':
        from magent.builtin.tf_model import DeepQNetwork
        models.append(DeepQNetwork(env, tiger_handle, "goal",
                                   batch_size=batch_size,
                                   memory_size=2 ** 20, target_update=2000, learning_rate=1e-4, reward_decay=0.99))
        step_batch_size = None

    elif args.alg == 'pytorch_dqn':
        from magent.builtin.pytorch_model import DeepQNetwork
        models.append(DeepQNetwork(env, tiger_handle, "goal",
                                   batch_size=batch_size,
                                   memory_size=2 ** 20, target_update=2000, learning_rate=1e-4, use_dueling=True))
        step_batch_size = None


    elif args.alg == 'drqn':
        from magent.builtin.tf_model import DeepRecurrentQNetwork
        models.append(DeepRecurrentQNetwork(env, tiger_handle, "goal",
                                            batch_size=int(batch_size / unroll), unroll_step=unroll,
                                            memory_size=20000, learning_rate=4e-4))
    elif args.alg == 'a2c':
        from magent.builtin.mx_model import AdvantageActorCritic

        step_batch_size = int(10 * args.map_size * args.map_size * 0.01)
        models.append(AdvantageActorCritic(env, tiger_handle, "goal",
                                           batch_size=step_batch_size,
                                           learning_rate=1e-2))

        step_batch_size = None
    else:
        raise NotImplementedError

    # load if
    savedir = 'save_model'
    if args.load_from is not None:
        start_from = args.load_from
        print("load ... %d" % start_from)
        for model in models:
            model.load(savedir, start_from)
    else:
        start_from = 0

    # init logger
    magent.utility.init_logger(args.name)

    # print debug info
    print(args)
    print("view_size", env.get_view_space(tiger_handle))

    # play
    train_id = 1 if args.train else -1
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        eps = magent.utility.linear_decay(int(k/100), 20, 0.05) if not args.greedy else 0
        loss, reward, value = play_a_round(env, args.map_size, [deer_handle, tiger_handle], models,
                                           agent_generator=args.agent_generator,
                                           step_batch_size=step_batch_size, train_id=train_id,
                                           print_every=40, render=args.render,
                                           eps=eps)

        log.info("round %d\t loss: %s\t reward: %s\t value: %s" % (k, loss, reward, value))
        print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

        if (k + 1) % args.save_every == 0:
            print("save model... ")
            for model in models:
                model.save(savedir, k)
