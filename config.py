from customer.base_customer import Customer
from customer.myopic import Myopic_Customer
from customer.seasonal import Seasonal_Customer
from customer.anticipating import Anticipating_Customer

# WANDB
project_name = "Master Thesis" # DO NOT CHANGE
run_name = "Try evaluation statistics"
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
number_of_training_episodes = 100000
number_of_simulation_episodes = 1

# Market
number_of_customers = 100
number_of_vendors = 1

# Customer
reference_price = 5
λ = 4
nothing_preference = 1
seasonal_reference_prices = [3, 5, 4, 5, 6, 7, 5] # mean = 5
# pricing functions: https://www.geogebra.org/graphing/kesahyyb

# Customer setup (mix must sum to 1)
customers: list[Customer] = [Seasonal_Customer(), Anticipating_Customer()]
customer_mix = [0.9, 0.1]

# Vendor
# Until now only using one single (monopolistic) vendor that is represented by the agent

# State Space
week_length = 7
max_waiting_pool = 100

# Action Space
max_price = 10.0
support_continuous_action_space = True

logged_config = {
    "rl_policy" : rl_policy,
    "gamma" : gamma,
    "constant_learning_rate" : constant_learning_rate,
    "initial_learning_rate" : initial_learning_rate,
    "episode_length" : episode_length,
    "rl_algorithm" : rl_algorithm,
    "number_of_training_episodes" : number_of_training_episodes,
    "number_of_customers" : number_of_customers,
    "number_of_vendors" : number_of_vendors,
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