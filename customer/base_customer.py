import numpy as np
from abc import ABC, abstractmethod, abstractproperty

import config

class Customer(ABC):

    last_prices = []

    def __init__(self):
        self.name = "undefined"

    @abstractmethod
    def generate_purchase_probabilities_from_offer(self, state, action) -> np.array:
        raise NotImplementedError
    
    def calculate_weight(self, price, reference_price):
        return ( -config.λ * np.exp(price - reference_price) - price) / reference_price + config.λ
