# State Space
week_length = 4
max_waiting_pool = 1000
n_timesteps_saving = 5

# Action Space
max_price = 10
support_continuous_action_space = True

# Market
n_customers = 50
product_cost = 0

# Customers (pricing functions: https://www.geogebra.org/graphing/kesahyyb)
edgeworth = False
reference_price = 5
λ = 4
nothing_preference = 1
seasonal_reference_prices = [4, 6, 7, 3] # [3, 5, 4, 5, 6, 7, 5] # mean = 5

# Customer setup (mix must sum to 1)
customers = ['strategic']
customer_mix = [1]
linearly_changing_customers = False # scales linearly from [1, 0] to [0, 1]
stochastic_customers = True

# Anticipating Customer
n_timesteps_predicting = week_length - 1
n_lags = week_length

# Strategic Customer
train_strategic = True
vendor_model_path = '/Users/fabian/Developer/HPI/Masterarbeit/StrategicCustomerSimulation/results/02/02_Monopol_no_storage/model.zip'
strategic_enter_waiting_pool = True
strategic_model_path = '/Users/fabian/Developer/HPI/Masterarbeit/StrategicCustomerSimulation/results/07/07_Seaonal_Model/model.zip'

# Competitor
undercutting_competitor = False
undercutting_competitor_step = 1
undercutting_competitor_floor = 0
undercutting_competitor_ceiling = None

# Training & Simulation
episode_length = week_length * 10
rl_algorithm = 'ppo'
n_training_episodes = 250 / 2
n_simulation_episodes = 1

# RL Parameters
rl_policy = 'MlpPolicy'
gamma = 1
initial_learning_rate = 0.00003
constant_learning_rate = True # decrease learning rate linearly to 0 if constant_learning_rate = False

# Logging
project_name = "[dev] Strategic Customer Simulation"
run_name     = "07_Seasonal_Model_short"
run_notes    = ""
wandb_mode   = 'disabled'
save_summary = True
plot_dir     = f"./results/{run_name}/plots/"
tb_dir       = f"./tensorboard/{run_name}/"
summary_dir  = f"./results/{run_name}/"
summary_file = "summary.txt"
info_file    = "data.txt"

# Callbacks
peval_cb_n_episodes = n_training_episodes / 1000
early_stopping_cb_n_episodes  = n_training_episodes / 1000
early_stopping_cb_threshold  = 0.0005
early_stopping_cb_min_amount  = 0.9999 # 0.03
sim_cb_n_episodes = n_training_episodes / 50