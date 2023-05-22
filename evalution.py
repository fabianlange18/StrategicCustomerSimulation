import wandb
import numpy as np
import matplotlib.pyplot as plt

import config


def evaluate_model(infos):
    
    print("\nStatistics:")
    print(f'{"Property": >40}{"Mean": >10}{"Std": >12}{"Min": >10}{"Median": >13}{"Max": >13}\n')

    for key in infos.keys():
        stats = [np.mean(infos[key]), np.std(infos[key]), np.min(infos[key]), np.median(infos[key]), np.max(infos[key])]
        print(f'{key: >40}{stats[0]: >10.2f}{stats[1]: >12.2f}{stats[2]: >10.2f}{stats[3]: >13.2f}{stats[4]: >13.2f}')
    print()

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
