from typing import Tuple
import numpy as np

from customers._0_base_customer import Customer

from collections import deque

import config
from util.softmax import softmax

class Price_Aware_Customer(Customer):

    def __init__(self):
        self.name = "price_aware"
        self.ability_to_wait = True
        self.discount = 0.85
        self.last_prices = [deque([], maxlen=4) for _ in range(1 + config.undercutting_competitor)]

    def generate_purchase_probabilities_from_offer(self, state, action) -> Tuple[np.array, int]:

        weights = [config.nothing_preference]
        reference_price = config.seasonal_reference_prices[state[0]]

        if len(self.last_prices[0]) >= 4 and min(self.last_prices[0]) >= action[0]:
            weights.append(10)
            # price = action[0]
            # weight = self.calculate_weight(price, reference_price = reference_price)
            # weights.append(weight)
        else:
            weights.append(-10)
        
        if config.undercutting_competitor:
            if len(self.last_prices[1]) >= config.n_timesteps_saving and min(self.last_prices[1]) * 0.85 > action[1]:
                weights.append(10)
                # price = action[1]
                # weight = self.calculate_weight(price, reference_price = reference_price)
                # weights.append(weight)
            else:
                weights.append(-10)

        # Append the price to the stored prices
        [self.last_prices[i].append(action[i]) for i in range(1 + config.undercutting_competitor)]

        return softmax(np.array(weights)), min(self.last_prices[0])
