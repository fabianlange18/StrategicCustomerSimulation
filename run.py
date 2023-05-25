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
        config=config.logged_config,
        mode= config.mode
    )
    
    model = train_model()
    infos = simulate_policy(model)
    evaluate_model(infos, model)
    
    run.finish()


def parse_arguments():
    parser = argparse.ArgumentParser(description="A simulation tool for strategic customer behavior. Specify the market simulation in config.py")
    parser.add_argument('--name', "-n", type=str, help='Specify a name for your run that is displayed in W&B')
    parser.add_argument('--wandb_mode', "-w", type=str, help='Usage of W&B: online, offline or disabled')
    args = parser.parse_args()
    if args.name != None:
        config.run_name = args.name
    if args.wandb_mode != None:
        config.mode = args.wandb_mode
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == '__main__':
    parse_arguments()
    main()