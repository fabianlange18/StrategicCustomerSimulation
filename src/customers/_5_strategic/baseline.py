from gym import Env
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete

import numpy as np
from collections import defaultdict

import config

class StrategicCustomer(Env):

    def __init__(self):
        
        self.name = "strategic"
        self.ability_to_wait = True

        # Observation Space
        self.observations_array = [config.week_length]
        self.observations_array.append(2)
        self.last_prices = [config.max_price * 100 for _ in range(config.n_timesteps_saving)]
        self.observation_space = MultiDiscrete([*self.observations_array, *self.last_prices])

        # Action Space
        self.action_space = Discrete(2)

        # Prices
        self.prices = self.init_random_price_trajectory()

        self.step_counter = 0
        self.reset()


    def step(self, action, simulation_mode = False):

        # Receive Price offer
        price = self.prices[self.s[0]]

        # Arriving customers
        # n_customers = config.n_customers + self.s[1]
        n_customers = 0 + self.s[1]
        self.s[1] = 0

        reward = (config.max_price - price) * action * n_customers

        # Customers enter Waiting Pool if no buy
        self.s[1] = min(n_customers * (1 - action), config.max_waiting_pool - 1)

        info = defaultdict(int)
        info["i0_agent_offer_price"] = price
        info["i0_n_strategic_buy"] = action * n_customers
        info["i0_strategic_reward"] = price * action * n_customers
        info["i0_n_strategic"] = n_customers
        info["i0_total_reward"] = price * action * n_customers
        info["i1_total_reward"] = 0
        info["i0_n_strategic_waiting"] = self.s[1]
        info["i0_strategic_reference_price"] = price

        # Store last (own) prices in last state dimensions
        state_index = 2
        for _ in range(config.n_timesteps_saving - 1):
            self.s[state_index] = self.s[state_index+1]
            state_index += 1
        self.s[state_index] = min(price * 100, config.max_price * 100 - 1)

        # Update state
        self.s[0] += 1
        self.s[0] %= config.week_length
        self.step_counter += 1
        done = self.step_counter == config.episode_length # or action == 1

        return self.s, reward, done, info


    def reset(self):
        self.init_random_price_trajectory()
        self.s = np.array([*[0 for _ in [*self.observations_array, *self.last_prices]]])
        self.s[1] = 1
        return self.s


    def init_random_price_trajectory(self):
        random = np.random.randint(0, config.max_price, config.week_length)
        random[3] = 0
        return random