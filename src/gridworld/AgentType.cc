/**
 * \file AgentType.cc
 * \brief implementation of AgentType (mainly initialization)
 */

#include "AgentType.h"

namespace magent {
namespace gridworld {


#define AGENT_TYPE_SET_INT(name) \
    if (strequ(keys[i], #name)) {\
        name = (int)(values[i] + 0.5);\
        is_set = true;\
    }

#define AGENT_TYPE_SET_FLOAT(name) \
    if (strequ(keys[i], #name)) {\
        name = values[i];\
        is_set = true;\
    }

#define AGENT_TYPE_SET_BOOL(name)\
    if (strequ(keys[i], #name)) {\
        name = bool(int(values[i] + 0.5));\
        is_set = true;\
    }

AgentType::AgentType(int n, std::string name, const char **keys, float *values, bool turn_mode, bool infection_mode) {
    std::cout << "Begin Agent Type Initialiation" << std::endl;

    this->name = name;

    // default value
    attack_in_group = false;
    width = length = 1;
    speed = 1.0; hp = 1.0;

    view_radius = 1; view_angle = 360;
    attack_radius = 0; attack_angle = 0;

    hear_radius = speak_radius = 0.0f;
    speak_ability = 0;

    damage = trace = eat_ability = step_recover = kill_supply = food_supply = 0;

    attack_in_group = false; can_absorb = false;
    step_reward = kill_reward  = dead_penalty = attack_penalty = 0.0;
    can_absorb = false;
    infection_radius = 0.0;
    infection_probability = 0.0;
    bad_vaccine_penalty = 0.0;

    // init member vars from str (reflection)
    bool is_set;
    for (int i = 0; i < n; i++) {
        is_set = false;
        AGENT_TYPE_SET_INT(width);
        AGENT_TYPE_SET_INT(length);

        AGENT_TYPE_SET_FLOAT(speed);
        AGENT_TYPE_SET_FLOAT(hp);

        AGENT_TYPE_SET_FLOAT(view_radius);  AGENT_TYPE_SET_FLOAT(view_angle);
        AGENT_TYPE_SET_FLOAT(attack_radius);AGENT_TYPE_SET_FLOAT(attack_angle);
        AGENT_TYPE_SET_FLOAT(vaccine_radius);AGENT_TYPE_SET_FLOAT(vaccine_angle);

        AGENT_TYPE_SET_FLOAT(hear_radius);  AGENT_TYPE_SET_FLOAT(speak_radius);
        AGENT_TYPE_SET_INT(speak_ability);

        AGENT_TYPE_SET_FLOAT(damage);       AGENT_TYPE_SET_FLOAT(trace);
        AGENT_TYPE_SET_FLOAT(eat_ability);
        AGENT_TYPE_SET_FLOAT(step_recover); AGENT_TYPE_SET_FLOAT(kill_supply);
        AGENT_TYPE_SET_FLOAT(food_supply);

        AGENT_TYPE_SET_BOOL(attack_in_group); AGENT_TYPE_SET_BOOL(can_absorb);

        AGENT_TYPE_SET_FLOAT(step_reward);  AGENT_TYPE_SET_FLOAT(kill_reward);
        AGENT_TYPE_SET_FLOAT(dead_penalty); AGENT_TYPE_SET_FLOAT(attack_penalty);

        AGENT_TYPE_SET_FLOAT(view_x_offset); AGENT_TYPE_SET_FLOAT(view_y_offset);
        AGENT_TYPE_SET_FLOAT(att_x_offset);  AGENT_TYPE_SET_FLOAT(att_y_offset);
        AGENT_TYPE_SET_FLOAT(turn_x_offset); AGENT_TYPE_SET_FLOAT(turn_y_offset);

        AGENT_TYPE_SET_FLOAT(infection_radius);
        AGENT_TYPE_SET_FLOAT(infection_probability);

        AGENT_TYPE_SET_FLOAT(vaccine_reward);

        AGENT_TYPE_SET_FLOAT(bad_vaccine_penalty);

        if (!is_set) {
            LOG(FATAL) << "invalid agent config in AgentType::AgentType : " << keys[i];
        }
    }

    std::cout << "Attributes loaded" << std::endl;


    // NOTE: do not support SectorRange with angle >= 180, only support circle range when angle >= 180
    int parity = width % 2; // use parity to make range center-symmetric
    if (view_angle >= 180) {
        if (fabs(view_angle - 360) > 1e-5) {
            LOG(FATAL) << "only supports ranges with angle = 360, when angle > 180.";
        }
        view_range = new CircleRange(view_radius, 0, parity);
    } else {
        view_range = new SectorRange(view_angle, view_radius, parity);
    }

    if (attack_angle >= 180) {
        if (fabs(attack_angle - 360) > 1e-5) {
            LOG(FATAL) << "only supports ranges with angle = 360, when angle > 180.";
        }
        attack_range = new CircleRange(attack_radius, width / 2.0f, parity);
    } else {
        attack_range = new SectorRange(attack_angle, attack_radius, parity);
    }

    if (infection_mode) {
        if (vaccine_angle >= 180) {
            if (fabs(vaccine_angle - 360) > 1e-5) {
                LOG(FATAL) << "only supports ranges with angle = 360, when angle > 180.";
            }
            vaccine_range = new CircleRange(vaccine_radius, width / 2.0f, parity);
        } else {
            vaccine_range = new SectorRange(vaccine_angle, vaccine_radius, parity);
        }
    }

    move_range   = new CircleRange(speed, 0, 1);
    view_x_offset = width / 2; view_y_offset = length / 2;
    att_x_offset  = width / 2; att_y_offset  = length / 2;
    vacc_x_offset  = width / 2; vacc_y_offset  = length / 2;
    turn_x_offset = 0; turn_y_offset = 0;

    move_base = 0;
    turn_base = move_range->get_count();

    if (turn_mode) {
        std::cout << "############ We're in turn mode ! #########" << std::endl;
        attack_base = turn_base + 2;
    } else {
        attack_base = turn_base;
    }

    if (infection_mode) {
        std::cout << "############ Infection Mode ! #########" << std::endl;
        vaccine_base = attack_base + attack_range->get_count();
    }

    int n_action = attack_base + attack_range->get_count();
    if (infection_mode)
        n_action +=  vaccine_range->get_count();
    for (int i = 0; i < n_action; i++) {
        action_space.push_back(i);
    }
    std::cout << "Agent Type Initialiation OK" << std::endl;

//    infection_range = new CircleRange(infection_radius, width / 2.0f, parity);
//    infection_range = new CircleRange(infection_radius, 0, 0);
    // action space layout : move turn attack ...
}

} // namespace magent
} // namespace gridworld
