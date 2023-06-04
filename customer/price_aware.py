from typing import Tuple
import numpy as np
from .base_customer import Customer

from collections import deque

import config
from util.softmax import softmax

class Price_Aware_Customer(Customer):

    def __init__(self):
        self.name = "price_aware"
        self.ability_to_wait = True
        self.last_prices = [deque([], maxlen=config.n_timesteps_saving) for _ in range(config.n_vendors)]

    def generate_purchase_probabilities_from_offer(self, state, action) -> Tuple[np.array, int]:

        weights = [config.nothing_preference]

        if len(self.last_prices[0]) >= config.n_timesteps_saving and min(self.last_prices[0]) * 0.85 > action[0]:
            weights.append(10)
        else:
            weights.append(-10)

        # Append the price to the stored prices
        [self.last_prices[i].append(action[i]) for i in range(config.n_vendors)]

        return softmax(np.array(weights)), min(self.last_prices[0]) * 0.85
