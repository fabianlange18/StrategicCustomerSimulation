import numpy as np
from ._0_base_customer import Customer

from collections import deque

from stable_baselines3.ppo import PPO

import config
from util.softmax import softmax

class RL_based_Customer(Customer):

    def __init__(self):
        self.name = "rl_based"
        self.ability_to_wait = True
        self.last_prices = [deque([], maxlen=4) for _ in range(1 + config.undercutting_competitor)]

        self.model = PPO.load(config.strategic_model_path)

    def generate_purchase_probabilities_from_offer(self, state, action) -> np.array:
        
        decisions = [0]

        decisions.append(self.model.predict([state[0], 0, *state[2:]], deterministic=True)[0])

        if max(decisions[1:]) == 0:
            decisions[0] = 1


        [self.last_prices[i].append(action[i]) for i in range(1 + config.undercutting_competitor)]

        
        return np.array(decisions), 0