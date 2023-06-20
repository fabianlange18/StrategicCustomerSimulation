import os
import argparse

import sys
sys.path.append("..")

import wandb
import config as config

from training import train_model

from evaluation.evaluator import Evaluator


def main():
    
    evaluator = Evaluator()
    
    run = wandb.init(
        project=config.project_name,
        name=config.run_name,
        notes=config.run_notes,
        sync_tensorboard=True,
        mode=config.wandb_mode
    )

    train_model(evaluator)
    
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