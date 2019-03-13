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


def generate_map(env, map_size, handles, agent_generator):
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

    elif agent_generator == 'two_clusters':
        x = y = map_size / 2
        n_deers = 81
        deer_pos = []
        for i in range(int(-np.sqrt(n_deers) / 2) - 1, int(np.sqrt(n_deers) / 2)):
            for j in range(int(-np.sqrt(n_deers) / 2) - 1, int(np.sqrt(n_deers) / 2)):
                deer_pos.append((x + i, y + j))

        env.add_agents(handles[0], method="custom", pos=deer_pos)

        n_tigers = 10
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



    else:
        env.add_agents(handles[0], method="random", n=map_size*map_size*0.1)
        env.add_agents(handles[1], method="random", n=map_size*map_size*0.05)


def play_a_round(env, map_size, handles, models, print_every, agent_generator,
                 train_id=1, step_batch_size=None, render=False, eps=None):
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

            if i == 1 and n_step == 0:
                print(obs[1][0][0].shape)
                for j in range(6):
                    cv2.imwrite(f'obs_{j}.png', 255*obs[i][0][0][:,:,j])

        # simulate one step
        n_step += 1
        done = env.step()

        if n_step == 160:
            ended = True
        # sample
        reward = 0
        if train_id != -1:
            rewards = env.get_reward(handles[train_id])
            # print(sum(env.get_reward(handles[train_id])) / env.get_num(handles[train_id]))
            rewards += sum(env.get_reward(handles[0])) / env.get_num(handles[train_id])

            alives  = env.get_alive(handles[train_id])
            reward = sum(rewards)
            total_reward += reward
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
    parser.add_argument("--n_round", type=int, default=30000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--agent_generator", default='random_spread', choices=['random_spread',
                                                                               'random_clusters',
                                                                               'two_clusters'])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--map_size", type=int, default=80)
    parser.add_argument("--name", type=str, default="goal")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn', 'a2c'])
    args = parser.parse_args()

    # init the game
    print('Init Env')
    env = magent.GridWorld("agent_goal", map_size=args.map_size)
    print('Env initialized')
    env.set_render_dir("build/render")

    # two groups of animal
    deer_handle, tiger_handle = env.get_handles()

    # init two models
    models = [
        RandomActor(env, deer_handle, tiger_handle),
    ]

    batch_size = 512
    unroll     = 8

    if args.alg == 'dqn':
        from magent.builtin.tf_model import DeepQNetwork
        models.append(DeepQNetwork(env, tiger_handle, "goal",
                                   batch_size=batch_size,
                                   memory_size=2 ** 20, learning_rate=4e-4))
        step_batch_size = None


    elif args.alg == 'drqn':
        from magent.builtin.tf_model import DeepRecurrentQNetwork
        models.append(DeepRecurrentQNetwork(env, tiger_handle, "goal",
                                            batch_size=int(batch_size / unroll), unroll_step=unroll,
                                            memory_size=20000, learning_rate=4e-4))

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
        eps = magent.utility.linear_decay(k, 10, 0.05) if not args.greedy else 0
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
