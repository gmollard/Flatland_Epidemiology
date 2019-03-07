""" tigers eat deer to get health point and reward"""

import magent


def get_config(map_size):
    gw = magent.gridworld

    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"embedding_size": 10})

    deer = cfg.register_agent_type(
        "deer",
        {'width': 1, 'length': 1, 'hp': 2, 'speed': 0,
         'view_range': gw.CircleRange(1), 'attack_range': gw.CircleRange(0),
         'damage': 0, 'step_recover': 0.2,
         'food_supply': 0, 'kill_supply': 8, 'kill_reward': 100,
         'infection_radius': 1, 'infection_probability': 0.1
         })

    tiger = cfg.register_agent_type(
        "tiger",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 1,
         'view_range': gw.CircleRange(30), 'attack_range': gw.CircleRange(1),
         'damage': 0, 'step_recover': 0.0,
         'food_supply': 0, 'kill_supply': 0,
         'step_reward': -0.1, 'attack_penalty': 0.0,
         'infection_radius': 0, 'infection_probability': 0.0
         })

    cfg.set_infection_mode()

    deer_group  = cfg.add_group(deer, prop_infected=0.01)
    tiger_group = cfg.add_group(tiger, prop_infected=0.1)

    # a = gw.AgentSymbol(tiger_group, index='any')
    # b = gw.AgentSymbol(deer_group, index='any')

    # cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=1e8)
    # cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=[a,b], value=[10000, 10000])

    return cfg
