# source: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

import config as config
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback

from market.simulation import simulate_policy
from market.market import Market

class EarlyStoppingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, evaluator, n_steps, verbose=0):

        self.evaluator = evaluator

        self.n_steps = n_steps
        self.last_time_trigger = 0
        self.min_passes = int(config.n_training_episodes * config.episode_length / n_steps * config.early_stopping_cb_min_amount)

        self.rewards = []

        super(EarlyStoppingCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_step(self) -> bool:

        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        continue_training = True

        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps

            infos = simulate_policy(self.model, deterministic=True, prog_bar=False)
            reward = np.sum(infos[f'i0_total_reward'][int(config.episode_length/2):] + infos[f'i1_total_reward'][int(config.episode_length/2):])
            self.rewards.append(reward)
            os.system('cls' if os.name == 'nt' else 'clear')
            print(config.run_name)
            if len(self.rewards) > 10:
                print(f'Current reward: {round(self.rewards[-1], 2)}, before: {[round(element, 2) for element in self.rewards[-10:-1]]}')
                print(f'Threshold [{config.early_stopping_cb_threshold * 100} % of Mean]: {np.mean(self.rewards[-10:]) * config.early_stopping_cb_threshold}')
                print(f'Std: {np.std(self.rewards[-10:])}')

            if np.std(self.rewards[-10:]) < np.mean(self.rewards[-10:]) * config.early_stopping_cb_threshold:
                continue_training = False

            if self.min_passes != 0:
                self.min_passes -= 1
                continue_training = True

            if self.n_calls == config.n_training_episodes * config.episode_length:
                continue_training = False

            if not continue_training:
                self.evaluator.write_output(f"\nTraining aborted after {self.num_timesteps} training steps because of conversion")
            
        return continue_training
                
    def _on_training_end(self):

        pass
