from gym import Env
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete

import numpy as np

import config

class StrategicCustomer(Env):

    def __init__(self, vendor_model):

        # Init last prices Storage and Observation Space (no waiting pool)
        last_prices = [config.max_price * 100 for _ in range(config.n_timesteps_saving)]
        self.observation_space = MultiDiscrete([config.week_length, config.max_waiting_pool, *last_prices])

        # Init Action Space
        self.action_space = Discrete(config.n_customers)

        self.vendor_model = vendor_model

        self.step_counter = 0
        self.reset()


    def step(self, action):

        price = self.vendor_model.predict(self.s, deterministic=True)

        # Waiting customers should also be part of the action: Action Space arbitrary
        reward = (config.max_price - price) * action

        info = {}
        done = False
        return self.s, reward, done, info


    def reset(self):
        self.s = np.array([0, 0, *[0 for _ in range(config.n_timesteps_saving)]])
        return self.s
