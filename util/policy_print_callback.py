# source: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

import wandb
import config
from customer.base_customer import Customer
from util.calc_optimal_policy import calculate_optimal_policy_seasonal, calculate_expected_reward, calculate_difference

from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps

class PolicyCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0):
        self.optimal_prices, self.optimal_profits = calculate_optimal_policy_seasonal()

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

        actual_prices = [self.model.predict([s, 0], deterministic=True)[0][0] for s in range(config.week_length)]
        Customer.last_prices = actual_prices
        [wandb.log({f'Mean_State_{s}' : actual_price}, step=self.n_calls) for s, actual_price in enumerate(actual_prices)]
        expected_profits_per_customer = calculate_expected_reward(actual_prices)
        wandb.log({'Price_Diff': calculate_difference(actual_prices, self.optimal_prices)}, step=self.n_calls)
        wandb.log({'Profit_Diff': calculate_difference(expected_profits_per_customer, self.optimal_profits)}, step=self.n_calls)
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