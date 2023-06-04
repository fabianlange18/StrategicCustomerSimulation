import config
from market.market import Market
from stable_baselines3.a2c import A2C
from stable_baselines3.ddpg import DDPG
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3
from stable_baselines3.common.monitor import Monitor

from util.schedule import linear_schedule
from util.callbacks.policy_evaluation_callback import policy_evaluation_callback
from util.callbacks.simulation_plot_callback import simulation_plot_callback

from wandb.integration.sb3 import WandbCallback

def train_model():

        model = select_model()
        print(f"\nTraining {config.rl_algorithm} model: {config.n_training_episodes} training episodes with each {config.episode_length} steps")
        model.learn(config.episode_length * config.n_training_episodes, callback=[WandbCallback(), policy_evaluation_callback, simulation_plot_callback], progress_bar=True)
        
        return model


def select_model():
    
    e = Market()

    e = Monitor(e, './logs')

    learning_rate = get_learning_rate()
    
    if config.rl_algorithm.lower() == 'a2c':
        model = A2C(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log='./logs')
    elif config.rl_algorithm.lower() == 'ddpg':
        model = DDPG(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log='./logs')
    elif config.rl_algorithm.lower() == 'ppo':
        model = PPO(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log='./logs')
    elif config.rl_algorithm.lower() == 'sac':
        model = SAC(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log='./logs')
    elif config.rl_algorithm.lower() == 'td3':
        model = TD3(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log='./logs')
    else:
        print(f"No model: {config.rl_algorithm}")
    
    return model

def get_learning_rate():
    if config.constant_learning_rate:
        return config.initial_learning_rate
    else:
        return linear_schedule(config.initial_learning_rate)