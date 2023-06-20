import config
import numpy as np

import matplotlib.pyplot as plt

from util.setup_folder import setup_folder
from .calc_optimal_policy import calculate_mean_difference

from market.market import Market
from customers._2_seasonal import Seasonal_Customer

class Evaluator:

    folder_exists = False

    def __init__(self) -> None:

        self.customers = Market().customers

        if config.save_summary and not self.folder_exists:
            setup_folder()
            self.print_header()
            self.folder_exists = True

    def write_output(self, output):
        if config.save_summary:
            f = open(f'{config.summary_dir}{config.summary_file}', 'a')
            f.write(output)
            f.close()
        else:
            print(output)


    def print_header(self):
        self.write_output(f"Summary for Run {config.run_name}\n\n")
        [self.write_output(f'{config.customer_mix[i] * 100} % {customer.name}\n') for i, customer in enumerate(self.customers)]
        if any([isinstance(customer, Seasonal_Customer) for customer in self.customers]):
            self.write_output(f'\nReference Prices Seasonal Customer: {config.seasonal_reference_prices}\n\n')


    def print_simulation_statistics(self, infos):
        self.write_output("\n\n\nStatistics for one deterministic simulation episode\n\n")
        self.write_output(f'{"Property": >40}{"Sum": >10}{"Mean": >12}{"Std": >10}{"Min": >12}{"Median": >14}{"Max": >13}\n\n')

        for key in infos.keys():
            stats = [np.sum(infos[key]), np.mean(infos[key]), np.std(infos[key]), np.min(infos[key]), np.median(infos[key]), np.max(infos[key])]
            self.write_output(f'{key: >40}{stats[0]: >10.2f}{stats[1]: >12.2f}{stats[2]: >10.2f}{stats[3]: >13.2f}{stats[4]: >13.2f}{stats[5]: >13.2f}\n')

        for customer in self.customers:
            reward = np.sum(infos[f'{customer.name}_reward'])
            buy = np.sum(infos[f'n_{customer.name}_buy'])
            self.write_output(f"\nAverage Agent Sales Price {customer.name}: {round(reward / buy, 3)}\n")
        
        self.write_output(f"Average Offer Price Agent: {round(np.mean(infos['agent_offer_price']), 3)}\n")
        if config.undercutting_competitor:
            self.write_output(f"Average Offer Price Competitor: {round(np.mean(infos['competitor_offer_price']), 3)}\n")

    def plot_trajectories(self, infos, show = False, save = ""):

        x = np.arange(config.episode_length * config.n_simulation_episodes)

        # check whether a waiting pool exists
        waiting_pool = False
        for customer in self.customers:
            waiting_pool = customer.ability_to_wait or waiting_pool

        if waiting_pool:
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
            total_waiting = np.sum([infos[f'n_{customer.name}_waiting'] for customer in self.customers], axis=0)
            ax5.step(x, total_waiting, color='black', where='pre')
            ax5.set_title('Waiting Pool')
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
        

        total_profit = np.sum([infos[f'{customer.name}_reward'] for customer in self.customers], axis=0)
        total_buying_decisions = np.sum([infos[f'n_{customer.name}_buy'] for customer in self.customers], axis=0)
        total_customers = np.sum([infos[f'n_{customer.name}'] for customer in self.customers], axis=0)

        ax1.step(x, total_profit, color='black', label='total', where='pre')
        ax2.step(x, infos[f'agent_offer_price'], color='black', where='pre')
        if config.undercutting_competitor:
            ax2.step(x, infos[f'competitor_offer_price'], color='green', where='mid')
        ax3.step(x, total_buying_decisions, color='black', where='pre')
        ax4.step(x, total_customers, color='black', where='pre')
        
        ax1.set_title('Profit by Customer Types')
        ax2.set_title('References Price/Prediction per Customer and Action Price')
        ax3.set_title('Buying Decisions by Customer Types')
        ax4.set_title('Number of Customers per Type')



        for customer in self.customers:
            ax1.step(x, infos[f'{customer.name}_reward'], label=customer.name, where='pre')
            ax2.step(x, infos[f'{customer.name}_reference_price'], label=customer.name, where='pre')
            ax3.step(x, infos[f'n_{customer.name}_buy'], label=customer.name, where='pre')
            ax4.step(x, infos[f'n_{customer.name}'], label=customer.name, where='pre')
            if customer.ability_to_wait:
                ax5.step(x, infos[f'n_{customer.name}_waiting'], label=customer.name, where='pre')

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels)

        plt.subplots_adjust(hspace=0.4)
        
        if show:
            plt.show()

        if save != "":
            plt.savefig(save)

        plt.close()


    def plot_seasonal_diff(self, values, title):
        plt.plot(values)
        plt.title(title)
        plt.savefig(f"{config.plot_dir}0_Seasonal_{title}")
        plt.close()

    def plot_prices(self, prices):
        for s in range(config.week_length):
            y = np.array(prices[s]['mean'])
            std = np.array(prices[s]['std'])
            x = np.arange(len(y))
            plt.plot(x, y, 'k-')
            plt.fill_between(x, (y - std).clip(min=0), (y + std).clip(max=config.max_price), alpha=0.5)
            plt.title(f'Price for day {s}')
            plt.savefig(f'{config.plot_dir}prices/state_{s}')
            plt.close()

    def plot_rewards(self, rewards):
        plt.plot(rewards)
        plt.title("Rewards of the deterministic policy")
        plt.savefig(f'{config.summary_dir}rewards')
        plt.close()

    def print_seasonal_policy_stats(self, actual_prices, expected_profits_per_customer, perfect_prices, perfect_profits_per_customer):

        self.write_output('\nStatistics for Seasonal Customers:\n')

        # ATTENTION: This is not 100% correct since the other states are not always 0
        self.write_output(f"\nACTUAL (empty waiting pool, stochastic_customers = {config.stochastic_customers})\n")
        self.seasonal_policy_writer(actual_prices, expected_profits_per_customer)

        self.write_output("\nOPTIMAL\n")
        self.seasonal_policy_writer(perfect_prices, perfect_profits_per_customer)

        self.write_output("\nPERFORMANCE\n")
        self.write_output(f"Reaching {round(calculate_mean_difference(actual_prices, perfect_prices) * 100, 2)} % of optimal prices\n")
        self.write_output(f"Reaching {round(calculate_mean_difference(expected_profits_per_customer, perfect_profits_per_customer) * 100, 2)} % of optimal profits")


    def seasonal_policy_writer(self, prices, profits_per_customer):

        seasonal_index = config.customers.index('seasonal')
                
        profits_all_customers = np.multiply(profits_per_customer, config.n_customers * config.customer_mix[seasonal_index])

        week_rewards = np.sum(profits_all_customers)
        episode_rewards = week_rewards * int(config.episode_length / config.week_length)

        rounded_prices = [round(price, 2) for price in prices]
        rounded_profits_per_cust = [round(profit, 2) for profit in profits_per_customer]
        rounded_profits_all_cust = [round(profit, 2) for profit in profits_all_customers]
        rounded_optimal_week_rewards = round(week_rewards, 2)
        rounded_optimal_episode_rewards = round(episode_rewards, 2)

        output = f'Policy Statistics for Prices: {rounded_prices}\nProfits per seasonal Customer: {rounded_profits_per_cust}\nProfits all seasonal Customers: {rounded_profits_all_cust}\nReward per Week: {rounded_optimal_week_rewards}\nReward per Episode: {rounded_optimal_episode_rewards}\n'
        self.write_output(output)