import wandb
import config
import numpy as np
import matplotlib.pyplot as plt

from customer.seasonal import Seasonal_Customer

from util.calc_optimal_policy import calculate_optimal_policy_seasonal, print_policy_stats, calculate_expected_reward, calculate_difference

def evaluate_model(infos, model):
    print_simulation_statistics(infos)
    plot_trajectories(infos)
    if any([isinstance(customer, Seasonal_Customer) for customer in config.customers]):
        print_policy_statistics(model)


def print_simulation_statistics(infos):
    print("\nStatistics:")
    print(f'{"Property": >40}{"Sum": >10}{"Mean": >12}{"Std": >10}{"Min": >12}{"Median": >14}{"Max": >13}\n')

    for key in infos.keys():
        stats = [np.sum(infos[key]), np.mean(infos[key]), np.std(infos[key]), np.min(infos[key]), np.median(infos[key]), np.max(infos[key])]
        print(f'{key: >40}{stats[0]: >10.2f}{stats[1]: >12.2f}{stats[2]: >10.2f}{stats[3]: >13.2f}{stats[4]: >13.2f}{stats[5]: >13.2f}')
    print()


def plot_trajectories(infos):
    customers = config.customers

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)

    x = np.arange(config.episode_length * config.number_of_simulation_episodes)

    for customer in customers:
        ax1.step(x, infos[f'{customer.name}_reward'], label=customer.name, where='post')
        ax2.step(x, infos[f'{customer.name}_reference_price'], label=customer.name, where='post')
        ax3.step(x, infos[f'n_{customer.name}_buy'], label=customer.name, where='post')
        ax4.step(x, infos[f'n_{customer.name}'], label=customer.name, where='post')
        ax5.step(x, infos[f'{customer.name}_consumer_rent'], label=customer.name, where='post')
    
    total_profit = np.sum([infos[f'{customer.name}_reward'] for customer in customers], axis=0)
    total_buying_decisions = np.sum([infos[f'n_{customer.name}_buy'] for customer in customers], axis=0)
    total_customers = np.sum([infos[f'n_{customer.name}'] for customer in customers], axis=0)
    total_consumer_rent = np.sum([infos[f'{customer.name}_consumer_rent'] for customer in customers], axis=0)
    
    ax1.step(x, total_profit, color='black', label='total', where='post')
    ax2.step(x, infos[f'action'], color='black', where='post')
    ax3.step(x, total_buying_decisions, color='black', where='post')
    ax4.step(x, total_customers, color='black', where='post')
    ax5.step(x, total_consumer_rent, color='black', where='post')
    
    ax1.set_title('Profit by Customer Types')
    ax2.set_title('References Price/Prediction per Customer and Action Price')
    ax3.set_title('Buying Decisions by Customer Types')
    ax4.set_title('Number of Customers per Type')
    ax5.set_title('Consumer Rent per Customer Type')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.show()
    wandb.log({'simulation_summary': fig})


def print_policy_statistics(model):
    actual_prices = [model.predict([s, 0], deterministic=True)[0][0] for s in range(config.week_length)]
    expected_profits_per_customer = calculate_expected_reward(actual_prices)
    print("\nACTUAL")
    print_policy_stats(actual_prices, expected_profits_per_customer)
    
    print("\nOPTIMAL")
    optimal_prices, optimal_profits_per_customer = calculate_optimal_policy_seasonal()
    print_policy_stats(optimal_prices, optimal_profits_per_customer)

    print("\nPERFORMANCE")
    print(f"Prices: {round(calculate_difference(actual_prices, optimal_prices), 3)}")
    print(f"Profits: {round(calculate_difference(expected_profits_per_customer, optimal_profits_per_customer), 3)}\n\n\n\n")
