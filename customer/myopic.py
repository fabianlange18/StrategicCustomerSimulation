import numpy as np
from .base_customer import Customer

import config
from util.softmax import softmax

class Myopic_Customer(Customer):

    def __init__(self):
        self.name = "myopic"

    def generate_purchase_probabilities_from_offer(self, state, action) -> np.array:

        weights = [config.nothing_preference]

        for vendor_idx in range(action.size):
            price = action[vendor_idx]
            weight = self.calculate_weight(price, config.reference_price)
            weights.append(weight)
        
        return softmax(np.array(weights)), config.reference_price