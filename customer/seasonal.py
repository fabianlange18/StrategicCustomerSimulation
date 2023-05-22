from typing import Tuple
import numpy as np
from .base_customer import Customer

import config
from util.softmax import softmax

class Seasonal_Customer(Customer):

    def __init__(self):
        self.name = "seasonal"

    def generate_purchase_probabilities_from_offer(self, state, action) -> Tuple[np.array, int]:

        weights = [config.nothing_preference]

        reference_price = config.seasonal_reference_prices[state[0]]

        for vendor_idx in range(action.size):
            price = action[vendor_idx]
            weight = self.calculate_weight(price, reference_price = reference_price)
            weights.append(weight)
        
        return softmax(np.array(weights)), reference_price