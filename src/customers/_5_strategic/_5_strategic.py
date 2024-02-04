import wandb

from gym import Env
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete

from stable_baselines3.ppo import PPO

import numpy as np
from collections import defaultdict

import config

class StrategicCustomer(Env):

    def __init__(self):

        self.name = "strategic"
        self.ability_to_wait = True

        # Init last prices Storage and Observation Space
        self.observations_array = [config.week_length]
        if config.strategic_enter_waiting_pool:
            # self.observations_array.append(config.max_waiting_pool)
            self.observations_array.append(2)
        last_prices = [config.max_price * 100 for _ in range(config.n_timesteps_saving)]
        self.observation_space = MultiDiscrete([*self.observations_array, *last_prices])
        # self.observation_space = MultiDiscrete([*self.observations_array])

        # Init Action Space
        self.action_space = Discrete(2) #  Box(low=0.0, high=1.0, shape=(1,)) # Discrete(config.n_customers)

        self.vendor_model = PPO.load(config.vendor_model_path)
        # self.min_price = min([self.vendor_model.predict([s], deterministic=True)[0][0] for s in range(config.week_length)])

        self.step_counter = 0
        self.reset()


    def step(self, action, simulation_mode = False):
        # Vier letzte Preise random * 2
        # Kunde soll rausfinden, dass es Sinn macht, beim Minimum zu kaufen

        # action = action[0] if isinstance(action, np.ndarray) else action

        # assert self.vendor_model.observation_space == self.observation_space, 'Observation Spaces of Customer and Vendor are not similar'

        # Receive Price offer from vendor
        price = self.vendor_model.predict(self.s, deterministic=True)[0][0]

        # Arriving customers
        # n_customers = config.n_customers + self.s[1]
        n_customers = self.s[1]
        # self.s[1] = 0

        # All customers buy or do not buy
        reward = (config.max_price - price) * action * n_customers

        # Customers enter Waiting Pool if no buy
        self.s[1] = min(n_customers * (1 - action), config.max_waiting_pool - 1)

        info = defaultdict(int)
        info["i0_agent_offer_price"] = price
        info["i0_n_strategic"] = n_customers
        info["i0_n_strategic_buy"] = action * n_customers
        info["i0_strategic_reward"] = price * action * n_customers
        info["i0_total_reward"] = price * action * n_customers
        # info["i1_total_reward"] = 0
        # TODO: Fix - not correct but works for now
        info["i0_n_strategic_waiting"] = self.s[1]

        # not completely correct
        info["i0_strategic_reference_price"] = reward # / n_customers


        # Store last (own) prices in last state dimensions
        state_index = 2 # + config.strategic_enter_waiting_pool
        if config.n_timesteps_saving > 0:
            for _ in range(config.n_timesteps_saving - 1):
                self.s[state_index] = self.s[state_index+1]
                state_index += 1
            self.s[state_index] = min(price * 100, config.max_price * 100 - 1)


        wandb.log(info)

        # Update state
        self.s[0] += 1
        self.s[0] %= config.week_length
        self.step_counter += 1
        done = self.s[0] == config.episode_length or self.s[1] == 0

        return self.s, reward, done, info


    def reset(self):
        # self.s = np.array([*[0 for _ in self.observations_array], *[0 for _ in range(config.n_timesteps_saving)]])
        self.s = np.array([0, 1, *[0 for _ in range(config.n_timesteps_saving)]])
        # self.s = np.array([0, 1])
        return self.s
