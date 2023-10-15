import config as config
from market.market import Market
from stable_baselines3.a2c import A2C
from stable_baselines3.ddpg import DDPG
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3
from stable_baselines3.dqn import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

from util.schedule import linear_schedule

from evaluation.callbacks.seasonal_evaluation_callback import SeasonalEvaluationCallback
from evaluation.callbacks.policy_evaluation_callback import PolicyEvaluationCallback
from evaluation.callbacks.simulation_plot_callback import SimulationPlotCallback
from evaluation.callbacks.early_stopping_callback import EarlyStoppingCallback

from wandb.integration.sb3 import WandbCallback

def train_model(evaluator):

        model = select_model()

        callbacks = setup_callbacks(evaluator)
        
        print(f"\nTraining {config.rl_algorithm} model {config.run_name}: {config.n_training_episodes} training episodes with each {config.episode_length} steps")
        model.learn(config.episode_length * config.n_training_episodes, callback=callbacks, progress_bar=True)

        model.save(f'{config.summary_dir}model')


def setup_callbacks(evaluator):

    policy_evaluation_callback = PolicyEvaluationCallback(evaluator, n_steps=config.episode_length * config.peval_cb_n_episodes)
    simulation_plot_callback = SimulationPlotCallback(evaluator, n_steps=config.episode_length * config.sim_cb_n_episodes)
    early_stopping_callback = EarlyStoppingCallback(evaluator, n_steps=config.episode_length * config.early_stopping_cb_n_episodes)

    callbacks = [WandbCallback(), simulation_plot_callback, early_stopping_callback, policy_evaluation_callback]
    
    if config.customers.__contains__("seasonal"):
        seasonal_callback = SeasonalEvaluationCallback(evaluator, n_steps=config.episode_length * config.peval_cb_n_episodes)
        callbacks.append(seasonal_callback)
    
    callback_list = CallbackList(callbacks)

    return callback_list


def select_model():
    
    e = Market()

    e = Monitor(e, config.tb_dir)

    learning_rate = get_learning_rate()
    
    if config.rl_algorithm.lower() == 'a2c':
        model = A2C(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log=config.tb_dir)
    elif config.rl_algorithm.lower() == 'ddpg':
        model = DDPG(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log=config.tb_dir)
    elif config.rl_algorithm.lower() == 'dqn':
        model = DQN(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log=config.tb_dir)
    elif config.rl_algorithm.lower() == 'ppo':
        model = PPO(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log=config.tb_dir)
    elif config.rl_algorithm.lower() == 'sac':
        model = SAC(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log=config.tb_dir)
    elif config.rl_algorithm.lower() == 'td3':
        model = TD3(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log=config.tb_dir)
    else:
        print(f"No model: {config.rl_algorithm}")
        NotImplementedError()
    
    return model

def get_learning_rate():
    if config.constant_learning_rate:
        return config.initial_learning_rate
    else:
        return linear_schedule(config.initial_learning_rate)