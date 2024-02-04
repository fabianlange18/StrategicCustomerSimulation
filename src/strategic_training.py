import config
import wandb
from wandb.integration.sb3 import WandbCallback

from evaluation.evaluator import Evaluator
from training import get_learning_rate, setup_callbacks
from customers._5_strategic._5_strategic import StrategicCustomer

from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor


def train_strategic_customer(evaluator):

    e = StrategicCustomer()

    e = Monitor(e, config.tb_dir)

    learning_rate = get_learning_rate()

    model = PPO(policy=config.rl_policy, env=e, learning_rate=learning_rate, gamma=config.gamma, tensorboard_log=config.tb_dir)

    callbacks = setup_callbacks(evaluator)

    model.learn(config.episode_length * config.n_training_episodes, callback=WandbCallback(), progress_bar=True) #, callback=callbacks

    model.save(f'{config.summary_dir}model')


def main():

    evaluator = Evaluator()

    run = wandb.init(
        project=config.project_name,
        name=config.run_name,
        notes=config.run_notes,
        sync_tensorboard=True,
        mode=config.wandb_mode
    )

    train_strategic_customer(evaluator)
    run.finish()

if __name__ == '__main__':
    main()