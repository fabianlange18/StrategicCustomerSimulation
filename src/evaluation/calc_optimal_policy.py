import config as config
import math
import numpy as np

from scipy.stats import gmean
from scipy.optimize import minimize

from customers._2_seasonal import Seasonal_Customer

def calculate_optimal_policy_seasonal():
    optimal_prices = []
    optimal_profits_per_customer = []

    for reference_price in config.seasonal_reference_prices:

        def profit(price):
            weight = Seasonal_Customer().calculate_weight(price, reference_price)
            return - price * ( np.exp(weight) / (np.exp(weight) + math.e) )

        optimization_result = minimize(profit, x0=0)
        optimal_prices.append(optimization_result.x[0])
        optimal_profits_per_customer.append(-optimization_result.fun)

    return optimal_prices, optimal_profits_per_customer


def calculate_expected_reward(prices):

    expected_profits_per_customer = []

    for price, reference_price in zip(prices, config.seasonal_reference_prices):
        
        weight = Seasonal_Customer().calculate_weight(price, reference_price)
        profit = price * ( np.exp(weight) / (np.exp(weight) + math.e) )
        expected_profits_per_customer.append(profit)
    
    return expected_profits_per_customer


def calculate_mean_difference(actual, optimal):
    diff = np.divide(actual, optimal)
    return np.mean(diff)