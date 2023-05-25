from gym import Env, Space
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.box import Box

import wandb
import config
import numpy as np



class Market(Env):

    def __init__(self):
        self.observation_space = MultiDiscrete([config.week_length, config.max_waiting_pool]) # Discrete(self.week_length)
        self.action_space = Box(low=0.0, high=config.max_price, shape=(1,)) if config.support_continuous_action_space else Discrete(config.max_price)

        assert sum(config.customer_mix) == 1, "The proportions for setting up the customer mix must sum to 1."

        self.reset()


    def step(self, action, simulation_mode=False):
        
        info = {}
        info["action"] = action[0]
        info["state_day"] = self.s[0]
        info["state_pool"] = self.s[1]

        # TODO: Apply this for multiple types of customers in duopoly setting
        # customers_per_vendor_iteration = config.number_of_customers / config.number_of_vendors

        customer_arrivals = np.random.multinomial(config.number_of_customers, config.customer_mix)

        for i, customer in enumerate(config.customers):
            info[f"n_{customer.name}"] = customer_arrivals[i]

        reward = 0.0

        for i, customer in enumerate(config.customers):
            probability_distribution, reference_price = customer.generate_purchase_probabilities_from_offer(self.s, action)
            customer_decisions = np.random.multinomial(customer_arrivals[i], probability_distribution.tolist())
            info[f"n_{customer.name}_buy"] = customer_decisions[1]
            info[f"{customer.name}_reference_price"] = reference_price
            # TODO: consumer rent is not accurately defined but can be taken as a measure for now
            info[f"{customer.name}_consumer_rent"] = customer_decisions[1] * (config.max_price - action[0])
            
            # WITH STOCHASTIC CUSTOMERS:
            info[f"{customer.name}_reward"] = customer_decisions[1] * action[0]
            reward += customer_decisions[1] * action[0]
            
            # WITHOUT STOCHASTIC CUSTOMERS:
            # info[f"{customer.name}_reward"] = probability_distribution[1] * action[0] * customer_arrivals[i]
            # reward += probability_distribution[1] * action[0] * customer_arrivals[i]

        info["total_reward"] = reward

        self.s[0] += 1
        self.s[0] %= 7
        self.step_counter += 1

        done = self.step_counter == config.episode_length

        if not simulation_mode and self.step_counter % config.episode_length < config.week_length:
            wandb.log(info)

        return self.s, float(reward), done, info


    def reset(self):
        self.s = np.array([0, 0])
        self.step_counter = 0
        return self.s