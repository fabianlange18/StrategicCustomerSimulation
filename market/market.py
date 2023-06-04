from gym import Env, Space
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.box import Box

import wandb
import config
import numpy as np

class Market(Env):

    def __init__(self):
        self.n_waiting_types = sum([customer.ability_to_wait for customer in config.customers])
        waiting_pool = [config.max_waiting_pool for _ in range(self.n_waiting_types)]
        last_prices = [config.max_price * 100 for _ in range(config.n_lags)]
        self.observation_space = MultiDiscrete([config.week_length, *waiting_pool, *last_prices])
        self.action_space = Box(low=0.0, high=config.max_price, shape=(1,)) if config.support_continuous_action_space else Discrete(config.max_price)

        assert sum(config.customer_mix) == 1, "The proportions for setting up the customer mix must sum to 1."

        self.reset()


    def step(self, action, simulation_mode=False):
        
        info = {}
        info["action"] = action[0]
        info["state_day"] = self.s[0]

        # TODO: Apply this for multiple types of customers in duopoly setting
        # customers_per_vendor_iteration = config.number_of_customers / config.number_of_vendors

        state_index = 1
        
        customer_arrivals = self.simulate_customer_arrivals()

        for i, customer in enumerate(config.customers):
            info[f"n_{customer.name}"] = customer_arrivals[i]

        reward = 0.0

        for i, customer in enumerate(config.customers):

            if customer.ability_to_wait:
                customer_arrivals[i] += self.s[state_index]

            probability_distribution, reference_price = customer.generate_purchase_probabilities_from_offer(self.s, action)
            customer_decisions = np.random.multinomial(customer_arrivals[i], probability_distribution.tolist())
            info[f"n_{customer.name}_buy"] = customer_decisions[1]
            info[f"{customer.name}_reference_price"] = reference_price
            # TODO: consumer rent is not accurately defined but can be taken as a measure for now
            info[f"{customer.name}_consumer_rent"] = customer_decisions[1] * (config.max_price - action[0])
            
            # WITH STOCHASTIC CUSTOMERS
            if config.stochastic_customers:
                info[f"{customer.name}_reward"] = customer_decisions[1] * action[0]
                reward += customer_decisions[1] * action[0]
            
            # WITHOUT STOCHASTIC CUSTOMERS
            else:
                info[f"{customer.name}_reward"] = probability_distribution[1] * action[0] * customer_arrivals[i]
                reward += probability_distribution[1] * action[0] * customer_arrivals[i]

            # WAITING CUSTOMERS
            if customer.ability_to_wait:
                self.s[state_index] = min(customer_decisions[0], config.max_waiting_pool - 1)
                info[f"n_{customer.name}_waiting"] = self.s[state_index]
                state_index += 1

            # LAST PRICES STARTING AT STATE INDEX
        for _ in range(config.n_lags - 1):
            self.s[state_index] = self.s[state_index+1]
            state_index += 1
        self.s[state_index] = min(action[0] * 100, config.max_price * 100 -1)

        info["total_reward"] = reward

        self.s[0] += 1
        self.s[0] %= config.week_length
        self.step_counter += 1

        done = self.step_counter == config.episode_length

        if not simulation_mode and self.step_counter % config.episode_length < config.week_length:
            wandb.log(info)

        return self.s, float(reward), done, info

    def simulate_customer_arrivals(self):
        if config.stochastic_customers:
            return np.random.multinomial(config.n_customers, config.customer_mix)
        else:
            return np.multiply(config.customer_mix, config.n_customers)


    def reset(self):
        self.s = np.array([0, *[0 for _ in range(self.n_waiting_types + config.n_lags)]])
        self.step_counter = 0
        return self.s