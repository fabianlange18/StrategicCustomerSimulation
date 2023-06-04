import os
import wandb
import config
import numpy as np
import matplotlib.pyplot as plt

from customer.seasonal import Seasonal_Customer

from util.calc_optimal_policy import calculate_optimal_policy_seasonal, print_policy_stats, calculate_expected_reward, calculate_difference


def setup_results_folder():
    os.makedirs(config.summary_dir, exist_ok=True)
    os.makedirs(config.plot_dir, exist_ok=True)
    print_config()

def print_config():
    f = open(f'{config.summary_dir}{config.summary_file}', 'x')
    f.write(f"Summary for Run {config.run_name}\n\n")
    [f.write(f'{config.customer_mix[i] * 100} % {customer.name}\n') for i, customer in enumerate(config.customers)]
    if any([isinstance(customer, Seasonal_Customer) for customer in config.customers]):
        f.write(f'\nReference Prices Seasonal Customer: {config.seasonal_reference_prices}\n\n')
    f.close()

def evaluate_model(infos, model):
    print_simulation_statistics(infos)
    plot_trajectories(infos, show=True)
    if any([isinstance(customer, Seasonal_Customer) for customer in config.customers]):
        print_policy_statistics(model)


def print_simulation_statistics(infos):
    f = open(f'{config.summary_dir}{config.summary_file}', 'a')
    f.write("\nStatistics:\n")
    f.write(f'{"Property": >40}{"Sum": >10}{"Mean": >12}{"Std": >10}{"Min": >12}{"Median": >14}{"Max": >13}\n\n')

    for key in infos.keys():
        stats = [np.sum(infos[key]), np.mean(infos[key]), np.std(infos[key]), np.min(infos[key]), np.median(infos[key]), np.max(infos[key])]
        f.write(f'{key: >40}{stats[0]: >10.2f}{stats[1]: >12.2f}{stats[2]: >10.2f}{stats[3]: >13.2f}{stats[4]: >13.2f}{stats[5]: >13.2f}\n')
    f.close()


def plot_trajectories(infos, log = False, show = False, save = ""):
    customers = config.customers

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True)

    x = np.arange(config.episode_length * config.n_simulation_episodes)

    for customer in customers:
        ax1.step(x, infos[f'{customer.name}_reward'], label=customer.name, where='post')
        ax2.step(x, infos[f'{customer.name}_reference_price'], label=customer.name, where='post')
        ax3.step(x, infos[f'n_{customer.name}_buy'], label=customer.name, where='post')
        ax4.step(x, infos[f'n_{customer.name}'], label=customer.name, where='post')
        ax5.step(x, infos[f'{customer.name}_consumer_rent'], label=customer.name, where='post')
        if customer.ability_to_wait:
            ax6.step(x, infos[f'n_{customer.name}_waiting'], label=customer.name, where='post')
    
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
    ax6.set_title('Waiting Pool')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels)
    
    if log:
        wandb.log({'simulation_summary': fig})
    
    if show:
        plt.show()

    if save is not "":
        plt.savefig(save)

    plt.close()


def print_policy_statistics(model):
    f = open(f'{config.summary_dir}{config.summary_file}', 'a')
    f.write("\nStatistics for Seasonal Customers:\n")

    # ATTENTION: This is not 100% correct since the other states are not always 0
    n_waiting_types = sum([customer.ability_to_wait for customer in config.customers])
    state = [0 for _ in range(n_waiting_types + config.week_length)]

    actual_prices = [model.predict([s, *state], deterministic=True)[0][0] for s in range(config.week_length)]
    expected_profits_per_customer = calculate_expected_reward(actual_prices)
    f.write(f"\nACTUAL (empty waiting pool, stochastic_customers = {config.stochastic_customers})\n")
    f.close()
    print_policy_stats(actual_prices, expected_profits_per_customer)

    f = open(f'{config.summary_dir}{config.summary_file}', 'a')
    f.write("\nOPTIMAL\n")
    f.close()
    optimal_prices, optimal_profits_per_customer = calculate_optimal_policy_seasonal()
    print_policy_stats(optimal_prices, optimal_profits_per_customer)

    f = open(f'{config.summary_dir}{config.summary_file}', 'a')
    f.write("\nPERFORMANCE\n")
    f.write(f"Reaching {round(calculate_difference(actual_prices, optimal_prices) * 100, 2)} % of optimal prices\n")
    f.write(f"Reaching {round(calculate_difference(expected_profits_per_customer, optimal_profits_per_customer) * 100, 2)} % of optimal profits")
    f.close()