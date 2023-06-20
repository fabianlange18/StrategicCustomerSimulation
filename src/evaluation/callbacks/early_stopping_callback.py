# source: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

import config as config
import numpy as np
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
        self.min_passes = int(config.early_stopping_cb_n_episodes / 3)

        self.customers = Market().customers

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

        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps

            infos = simulate_policy(self.model, deterministic=True, prog_bar=False)
            reward = np.sum([infos[f'{customer.name}_reward'] for customer in self.customers])
            self.rewards.append(reward)

            if self.min_passes != 0:
                self.min_passes -= 1
                return True
            else:
                # somehow it does not work to insert the boolean state directly into return
                if np.std(self.rewards[-10:]) < 5:
                    self.evaluator.write_output(f"\nTraining aborted after {self.num_timesteps} training steps")
                    return False
