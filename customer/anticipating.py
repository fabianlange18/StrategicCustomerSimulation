from typing import Tuple
import numpy as np
from .base_customer import Customer

from collections import deque

from statsmodels.tsa.ar_model import AutoReg

import config
from util.softmax import softmax

class Anticipating_Customer(Customer):

    def __init__(self):
        self.name = "anticipating"
        self.ability_to_wait = True
        self.last_prices = [deque([], maxlen=config.n_timesteps_saving) for _ in range(config.n_vendors)]
        self.predictions = np.ndarray((config.n_vendors, config.n_timesteps_predicting))

    def generate_purchase_probabilities_from_offer(self, state, action) -> Tuple[np.array, int]:

        weights = [config.nothing_preference]

        if self.predict_next_prices() and action[0] < min(self.predictions[0]):

            reference_price = config.seasonal_reference_prices[state[0]]

            for vendor_idx in range(action.size):
                price = action[vendor_idx]
                weight = self.calculate_weight(price, reference_price = reference_price)
                weights.append(weight)
        
        else:
            weights.append(-10)

        # Append the price to the stored prices
        [self.last_prices[i].append(action[i]) for i in range(config.n_vendors)]

        return softmax(np.array(weights)), self.predictions[0][state[0]]
    
    def predict_next_prices(self):
        if len(self.last_prices[0]) == config.n_timesteps_saving:
            self.predictions = [
                AutoReg(list(self.last_prices[0]), lags = config.n_lags)
                .fit()
                .predict(start= len(self.last_prices[0]), end= len(self.last_prices[0]) + config.n_timesteps_predicting - 1)
                    for i in range(config.n_vendors)]
        return len(self.last_prices[0]) == config.n_timesteps_saving
