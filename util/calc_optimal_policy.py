import config
import math
import numpy as np

from scipy.stats import gmean
from scipy.optimize import minimize

from customer.seasonal import Seasonal_Customer

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

def calculate_difference(actual, optimal):
    diff = np.divide(actual, optimal)
    return np.mean(diff) #, gmean(diff)


def print_policy_stats(prices, profits_per_customer):

    seasonal_index = [isinstance(customer, Seasonal_Customer) for customer in config.customers].index(True)
    
    profits_all_customers = np.multiply(profits_per_customer, config.n_customers * config.customer_mix[seasonal_index])

    week_rewards = np.sum(profits_all_customers)
    episode_rewards = week_rewards * int(config.episode_length / config.week_length)

    rounded_prices = [round(price, 2) for price in prices]
    rounded_profits_per_cust = [round(profit, 2) for profit in profits_per_customer]
    rounded_profits_all_cust = [round(profit, 2) for profit in profits_all_customers]
    rounded_optimal_week_rewards = round(week_rewards, 2)
    rounded_optimal_episode_rewards = round(episode_rewards, 2)

    f = open(f'{config.summary_dir}{config.summary_file}', 'a')
    f.write(f'Policy Statistics for Prices: {rounded_prices}\n')
    f.write(f'Profits per Customer: {rounded_profits_per_cust}\n')
    f.write(f'Profits all Customers: {rounded_profits_all_cust}\n')
    f.write(f'Reward per Week: {rounded_optimal_week_rewards}\n')
    f.write(f'Reward per Episode: {rounded_optimal_episode_rewards}\n')
    f.close()