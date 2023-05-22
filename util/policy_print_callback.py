# source: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

import wandb
import config
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps

class PolicyCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0):
        super(PolicyCallback, self).__init__(verbose)
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

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:

        # print(f'\n\nTraining of {self.num_timesteps} timesteps done - Resulting Pricing Policy (empty waiting pool):')
        for s in range(config.week_length):
            predictions = []
            for _ in range(1000):
                predictions.append(self.model.predict([s, 0])[0][0])
            # print(f'State {s}: {np.mean(predictions): >3.2f} € (std: {np.std(predictions): >3.3f})')
            wandb.log({f'Mean_State_{s}' : np.mean(predictions)})
            wandb.log({f'Std_State_{s}' : np.std(predictions)})
        
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

inner_callback = PolicyCallback()

policy_callback = EveryNTimesteps(n_steps=config.episode_length * 10, callback=inner_callback)