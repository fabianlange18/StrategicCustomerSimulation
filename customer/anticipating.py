from typing import Tuple
import numpy as np
from .base_customer import Customer

from collections import deque

from statsmodels.tsa.ar_model import AutoReg

import config
from util.softmax import softmax

class Anticipating_Customer(Customer):

    NUMBER_OF_TIMESTEPS_SAVING = 25
    NUMBER_OF_TIMESTEPS_PREDICTING = 10
    NUMBER_OF_VENDORS = 1
    NUMBER_OF_LAGS = 7

    def __init__(self):
        self.name = "anticipating"
        self.last_prices = [deque([], maxlen=self.NUMBER_OF_TIMESTEPS_SAVING) for _ in range(self.NUMBER_OF_VENDORS)]
        self.predictions = np.ndarray((self.NUMBER_OF_TIMESTEPS_PREDICTING, self.NUMBER_OF_VENDORS))

    def generate_purchase_probabilities_from_offer(self, state, action) -> Tuple[np.array, int]:

        # Append the price to the stored prices
        [self.last_prices[i].append(action[i]) for i in range(self.NUMBER_OF_VENDORS)]

        if len(self.last_prices[0]) == self.NUMBER_OF_TIMESTEPS_SAVING:
            self.predictions = [
                AutoReg(list(self.last_prices[i]), lags = self.NUMBER_OF_LAGS)
                .fit()
                .predict(start= len(self.last_prices[i]), end= len(self.last_prices[i]) + self.NUMBER_OF_TIMESTEPS_PREDICTING)
                    for i in range(self.NUMBER_OF_VENDORS)]

        weights = [config.nothing_preference]

        reference_price = config.seasonal_reference_prices[state[0]]

        for vendor_idx in range(action.size):
            price = action[vendor_idx]
            weight = self.calculate_weight(price, reference_price = reference_price)
            weights.append(weight)
        
        return softmax(np.array(weights)), self.predictions[0][0]