from typing import Tuple
import numpy as np
from .base_customer import Customer

from collections import deque

from statsmodels.tsa.ar_model import AutoReg

import config
from util.softmax import softmax

class Anticipating_Customer(Customer):

    # NUMBER_OF_TIMESTEPS_SAVING = 25
    NUMBER_OF_TIMESTEPS_PREDICTING = 7
    NUMBER_OF_LAGS = 7

    def __init__(self):
        self.name = "anticipating"
        # self.last_prices = [deque([], maxlen=self.NUMBER_OF_TIMESTEPS_SAVING) for _ in range(self.NUMBER_OF_VENDORS)]
        self.predictions = np.ndarray((config.number_of_vendors, self.NUMBER_OF_TIMESTEPS_PREDICTING))

    def generate_purchase_probabilities_from_offer(self, state, action) -> Tuple[np.array, int]:

        # Append the price to the stored prices
        # [self.last_prices[i].append(action[i]) for i in range(self.NUMBER_OF_VENDORS)]

        weights = [config.nothing_preference]

        if self.predict_next_prices() and state[0] == np.argmin(self.predictions[0]):

            reference_price = config.seasonal_reference_prices[state[0]]

            for vendor_idx in range(action.size):
                price = action[vendor_idx]
                weight = self.calculate_weight(price, reference_price = reference_price)
                weights.append(weight)
        
        else:
            weights.append(-10)

        return softmax(np.array(weights)), self.predictions[0][state[0]]
    
    def predict_next_prices(self):
        if len(self.last_prices) == config.week_length:
            # self.predictions = [
            #     AutoReg(list(self.last_prices) * 3, lags = self.NUMBER_OF_LAGS)
            #     .fit()
            #     .predict(start= len(self.last_prices), end= len(self.last_prices) + self.NUMBER_OF_TIMESTEPS_PREDICTING - 1)
            #         for i in range(config.number_of_vendors)]
            self.predictions = [self.last_prices]
        return len(self.last_prices) == config.week_length
