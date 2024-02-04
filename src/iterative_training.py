import sys
sys.path.append("..")

import wandb
import config
from util.update_config_dirs import update_dirs

from training import train_model
from evaluation.evaluator import Evaluator


import config
import wandb
from wandb.integration.sb3 import WandbCallback

from evaluation.evaluator import Evaluator
from training import get_learning_rate, setup_callbacks
from customers._5_strategic._5_strategic import StrategicCustomer

from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor


def train_vendor(model):
    
    evaluator = Evaluator()
    
    run = wandb.init(
        project=config.project_name,
        name=config.run_name,
        notes=config.run_notes,
        sync_tensorboard=True,
        mode=config.wandb_mode
    )

    train_model(evaluator, model)
    run.finish()


def train_strategic_customer(evaluator, model):

    e = StrategicCustomer()

    e = Monitor(e, config.tb_dir)

    learning_rate = get_learning_rate()

    if model == None:
        model = PPO(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log=config.tb_dir)
    else:
        model.set_env(e)

    callbacks = setup_callbacks(evaluator)

    model.learn(config.episode_length * config.n_training_episodes, callback=WandbCallback(), progress_bar=True) #, callback=callbacks

    model.save(f'{config.summary_dir}model')


def train_customer(model):

    evaluator = Evaluator()

    run = wandb.init(
        project=config.project_name,
        name=config.run_name,
        notes=config.run_notes,
        sync_tensorboard=True,
        mode=config.wandb_mode
    )

    train_strategic_customer(evaluator, model)
    run.finish()



def main():

    iterations = 20
    vendor_episodes = 10000
    customer_episodes = 1000


    for i in range(iterations):

        config.customers = ['seasonal', 'rl_based']
        config.customer_mix = [0.9, 0.1]
        config.train_strategic = False
        config.n_training_episodes = vendor_episodes
        config.run_name = f'07_Seasonal_RL_Market_long_{i * vendor_episodes}'
        update_dirs()

        if i == 0:
            config.customers = ['seasonal', 'recurring']
            config.customer_mix = [1, 0.0]
            model = None
        else:
            config.strategic_model_path = f'/Users/fabian/Developer/HPI/Masterarbeit/StrategicCustomerSimulation/results/07_Seasonal_Model_long_{(i - 1) * customer_episodes}/model'
            model = PPO.load(f'/Users/fabian/Developer/HPI/Masterarbeit/StrategicCustomerSimulation/results/07_Seasonal_RL_Market_long_{(i - 1) * vendor_episodes}/model')

        train_vendor(model)

        config.customers = ['strategic']
        config.customer_mix = [1]
        config.train_strategic = True
        config.n_training_episodes = customer_episodes
        config.run_name = f'07_Seasonal_Model_long_{i * customer_episodes}'
        config.vendor_model_path = f'/Users/fabian/Developer/HPI/Masterarbeit/StrategicCustomerSimulation/results/07_Seasonal_RL_Market_long_{i * vendor_episodes}/model'
        update_dirs()

        if i == 0:
            model = None
        else:
            model = PPO.load(f'/Users/fabian/Developer/HPI/Masterarbeit/StrategicCustomerSimulation/results/07_Seasonal_Model_long_{(i - 1) * customer_episodes}/model')


        train_customer(model)





if __name__ == '__main__':
    main()