from customer.base_customer import Customer
from customer.myopic import Myopic_Customer
from customer.seasonal import Seasonal_Customer
from customer.price_aware import Price_Aware_Customer
from customer.anticipating import Anticipating_Customer

# WANDB
project_name = "[dev] Strategic Customer Simulation" # DO NOT CHANGE
run_name = "Reproduce_Price_Aware"
run_notes = ""
mode = 'disabled'

# RL Parameters
rl_policy = 'MlpPolicy'
gamma = 1 # Do not discount for now
constant_learning_rate = True # did not turn out to be successful for now
initial_learning_rate = 0.00003 # decreases linearly to 0 if constant_learning_rate = False

# Training & Simulation
episode_length = 70
rl_algorithm = 'ppo'
n_training_episodes = 150000
n_simulation_episodes = 1

# Callbacks
peval_cb_n_episodes = 10000
plot_cb_n_episodes = 10000

# Market
n_customers = 50
n_vendors = 1

# Customer
reference_price = 5
λ = 4
nothing_preference = 1
seasonal_reference_prices = [3, 5, 4, 5, 6, 7, 5] # mean = 5
# pricing functions: https://www.geogebra.org/graphing/kesahyyb

# Customer setup (mix must sum to 1)
customers: list[Customer] = [Price_Aware_Customer] #[Seasonal_Customer, Anticipating_Customer]
customer_mix = [1]
stochastic_customers = True

# Anticipating Customer
n_timesteps_saving = 5
n_timesteps_predicting = 7
n_lags = 7

# Vendor
# Until now only using one single (monopolistic) vendor that is represented by the agent

# State Space
week_length = 7
max_waiting_pool = 1000

# Action Space
max_price = 10.0
support_continuous_action_space = True

# Results
plot_dir = f"./results/{run_name}/plots/"
summary_dir = f"./results/{run_name}/"
summary_file = "summary.txt"

logged_config = {
    "rl_policy" : rl_policy,
    "gamma" : gamma,
    "constant_learning_rate" : constant_learning_rate,
    "initial_learning_rate" : initial_learning_rate,
    "episode_length" : episode_length,
    "rl_algorithm" : rl_algorithm,
    "n_training_episodes" : n_training_episodes,
    "n_customers" : n_customers,
    "n_vendors" : n_vendors,
    "reference_price" : reference_price,
    "λ" : λ,
    "nothing_preference" : nothing_preference,
    "seasonal_reference_prices" : seasonal_reference_prices,
    "customers" : customers,
    "customer_mix" : customer_mix,
    "week_length" : week_length,
    "max_waiting_pool" : max_waiting_pool,
    "max_price" : max_price,
    "support_continuous_action_space" : support_continuous_action_space
}