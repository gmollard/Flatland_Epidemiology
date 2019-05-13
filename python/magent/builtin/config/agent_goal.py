""" tigers eat deer to get health point and reward"""

import magent



def get_config(map_size, vaccine_reward=1, view_radius=15, step_reward=-0.01, bad_vaccine_penalty=-0.1,
               collide_penalty=-0.1, infection_prob=0.02):
    gw = magent.gridworld

    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"embedding_size": 10})
    cfg.set({"infection_mode": True})

    deer = cfg.register_agent_type(
        "deer",
        {'width': 1, 'length': 1, 'hp': 2, 'speed': 0,
         'view_range': gw.CircleRange(1), 'attack_range': gw.CircleRange(0),
         'damage': 0, 'step_recover': 0.0,
         'food_supply': 0, 'kill_supply': 8, 'kill_reward': 0, 'vaccine_range': gw.CircleRange(0),
         'infection_radius': 2, 'infection_probability': infection_prob
         })

    tiger = cfg.register_agent_type(
        "tiger",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
         'view_range': gw.CircleRange(view_radius), 'attack_range': gw.CircleRange(0), 'vaccine_range': gw.CircleRange(1),
         'damage': 1, 'step_recover': 0.0,
         'food_supply': 0, 'kill_supply': 0,
         'step_reward': step_reward, 'attack_penalty': 0.0, "vaccine_reward": vaccine_reward,
         "bad_vaccine_penalty": bad_vaccine_penalty
         # 'infection_radius': 2, 'infection_probability': 0.1
         })

    cfg.set_infection_mode()


    deer_group  = cfg.add_group(deer, prop_infected=0.00)
    tiger_group = cfg.add_group(tiger, prop_infected=0)


    print('config done')

    a = gw.AgentSymbol(tiger_group, index='any')
    b = gw.AgentSymbol(deer_group, index='any')

    # cfg.add_reward_rule(gw.Event(b, 'infected'), receiver=[b], value=[-10])
    cfg.add_reward_rule(gw.Event(a, 'collide', b), receiver=[a], value=[collide_penalty])
    cfg.add_reward_rule(gw.Event(a, 'vaccine', b), receiver=[a], value=[1])
    # print('ok')
    # cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=[a,b], value=[10000, 10000])

    return cfg
