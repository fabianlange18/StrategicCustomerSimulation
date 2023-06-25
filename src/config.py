# State Space
week_length = 3
max_waiting_pool = 1000
n_timesteps_saving = 0 # 3 * week_length

# Action Space
max_price = 10.0
support_continuous_action_space = True

# Market
n_customers = 50

# Customers (pricing functions: https://www.geogebra.org/graphing/kesahyyb)
reference_price = 5
λ = 4
nothing_preference = 1
seasonal_reference_prices = [2, 4, 6] # [3, 5, 4, 5, 6, 7, 5] # mean = 5

# Customer setup (mix must sum to 1)
customers = ['seasonal']
customer_mix = [1]
linearly_changing_customers = False # scales linearly from [1, 0] to [0, 1]
stochastic_customers = False

# Anticipating Customer
n_timesteps_predicting = week_length - 1
n_lags = week_length

# Competitor
undercutting_competitor = True
undercutting_competitor_step = 1
undercutting_competitor_floor = 1
undercutting_competitor_ceiling = None

# Training & Simulation
episode_length = week_length * 10
rl_algorithm = 'ppo'
n_training_episodes = 100000
n_simulation_episodes = 1

# RL Parameters
rl_policy = 'MlpPolicy'
gamma = 1
initial_learning_rate = 0.00003
constant_learning_rate = True # decrease learning rate linearly to 0 if constant_learning_rate = False

# Logging
project_name = "[dev] Strategic Customer Simulation"
run_name     = "Seasonal_Duopol_3week"
run_notes    = ""
wandb_mode   = 'disabled'
save_summary = True
plot_dir     = f"./results/{run_name}/plots/"
tb_dir       = f"./tensorboard/{run_name}/"
summary_dir  = f"./results/{run_name}/"
summary_file = "summary.txt"

# Callbacks
peval_cb_n_episodes = n_training_episodes / 1000
early_stopping_cb_n_episodes  = n_training_episodes / 1000
sim_cb_n_episodes = n_training_episodes / 100