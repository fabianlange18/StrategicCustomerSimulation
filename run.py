import os
import argparse

import wandb
import config

from training import train_model
from simulation import simulate_policy
from evalution import evaluate_model

def main():
    run = wandb.init(
        project=config.project_name,
        name=config.run_name,
        notes=config.run_notes,
        sync_tensorboard=True,
        config=config.logged_config
    )
    model = train_model()
    infos = simulate_policy(model)
    evaluate_model(infos)
    run.finish()

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    parser = argparse.ArgumentParser(description="A simulation tool for strategic customer behavior. Specify the market simulation in config.py")
    parser.add_argument('--name', "-n", type=str, help='Specify a name for your run that is displayed in W&B')
    args = parser.parse_args()
    if args.name != None:
        config.run_name = args.name
    main()