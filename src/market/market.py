from gym import Env
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.box import Box

import wandb
import config
import numpy as np
from collections import defaultdict

from customers._1_myopic import Myopic_Customer as myopic
from customers._2_seasonal import Seasonal_Customer as seasonal
from customers._3_price_aware import Price_Aware_Customer as price_aware
from customers._4_anticipating import Anticipating_Customer as anticipating
from customers._5_strategic._5_strategic import StrategicCustomer as strategic

from .undercutting_vendor import Undercutting_Vendor

class Market(Env):

    def __init__(self):

        # Init Customers
        self.customers = self.init_customers()
        
        # Init Waiting Pool
        self.n_waiting_types = sum([customer.ability_to_wait for customer in self.customers])
        waiting_pool = [config.max_waiting_pool for _ in range(self.n_waiting_types)]

        # Init last prices Storage and Observation Space
        last_prices = [config.max_price * 100 for _ in range(config.n_timesteps_saving)]
        self.observation_space = MultiDiscrete([config.week_length, *waiting_pool, *last_prices])

        # Init Action Space
        self.action_space = Box(low=0.0, high=config.max_price, shape=(1,)) if config.support_continuous_action_space else Discrete(config.max_price)

        # Init Competitor
        if config.undercutting_competitor:
            self.competitor = Undercutting_Vendor()
        else:
            self.competitor = None


        self.step_counter = 0
        self.reset()


    def step(self, action, simulation_mode=False):

        if isinstance(action, np.int64):
            action = np.array([action])

        reward = [0.0, 0.0]
        competitor_profit = [0.0, 0.0]

        # Simulate customer arrivals
        customer_arrivals = self.simulate_customer_arrivals(simulation_mode)

        # Add waiting customers
        state_index = 1
        for i, customer in enumerate(self.customers):
            if customer.ability_to_wait:
                customer_arrivals[i] += self.s[state_index]
                self.s[state_index] = 0
                state_index += 1
        
        # Init Logging
        info = defaultdict(int)

        # Include competitor offer
        if self.competitor is not None:
            action = np.append(action, self.competitor.price)
            info["i0_competitor_offer_price"] = action[1]

        # Logging
        info["i0_agent_offer_price"] = action[0]
        info["i1_agent_offer_price"] = action[0]
        
        # Devide customer arrivals by 2 if there is a competitor
        customer_arrivals = customer_arrivals / (1 + config.undercutting_competitor)
        
        for i, customer in enumerate(self.customers):
            info[f"i0_n_{customer.name}"] = customer_arrivals[i]
            info[f"i1_n_{customer.name}"] = customer_arrivals[i]

        # Vendor iterations
        for i in range(1 + config.undercutting_competitor):
            
            state_index = 1

            if i == 1:
                # Update Price of Competitor after the first iteration
                if config.undercutting_competitor:
                    action = np.array([action[0], self.competitor.update_price(action[0])])
                    info['i1_competitor_offer_price'] = action[1]

            # Simulate every customer
            for j, customer in enumerate(self.customers):

                # Calculate purchase probabilities first iteration
                probability_distribution, reference_price = customer.generate_purchase_probabilities_from_offer(self.s, action)

                # Simulate customer decisions
                if config.stochastic_customers:
                    customer_decisions = np.random.multinomial(customer_arrivals[j], probability_distribution.tolist())
                else:
                    customer_decisions = probability_distribution * customer_arrivals[j]

                # Calculate reward
                customer_reward = customer_decisions[1] * (action[0] - config.product_cost)

                if config.undercutting_competitor:
                    customer_competitor_profit = customer_decisions[2] * (action[1] - config.product_cost)
                    competitor_profit[i] += customer_competitor_profit
                reward[i] += customer_reward

                # Logging
                info[f"i{i}_n_{customer.name}_buy"] = customer_decisions[1]
                if config.undercutting_competitor:
                    info[f"i{i}_n_{customer.name}_competitor_buy"] = customer_decisions[2]
                    info[f"i{i}_{customer.name}_competitor_reward"] = customer_competitor_profit
                info[f"i{i}_{customer.name}_reference_price"] = reference_price
                info[f"i{i}_{customer.name}_reward"] = customer_reward

                # Not buying customers enter waiting pool
                if customer.ability_to_wait:
                    self.s[state_index] += customer_decisions[0]
                    self.s[state_index] = min(self.s[state_index], config.max_waiting_pool - 1)
                    info[f"i{i}_n_{customer.name}_waiting"] += self.s[state_index]
                    state_index += 1


        # Store last (own) prices in last state dimensions
        if config.n_timesteps_saving > 0:
            for _ in range(config.n_timesteps_saving - 1):
                self.s[state_index] = self.s[state_index+1]
                state_index += 1
            self.s[state_index] = min(action[0] * 100, config.max_price * 100 - 1)

        # Update state
        self.s[0] += 1
        self.s[0] %= config.week_length
        self.step_counter += 1
        done = self.step_counter == config.episode_length

        # Logging
        info["i0_total_reward"], info["i1_total_reward"] = reward[0], reward[1]
        info["i0_total_competitor_reward"], info["i1_total_competitor_reward"] = competitor_profit[0], competitor_profit[1]
        if not simulation_mode and self.s[0] % config.episode_length < config.week_length:
            wandb.log(info)

        return self.s, float(sum(reward)), done, info #False, 


    def init_customers(self):
        customers = [globals()[class_name] for class_name in config.customers]
        customers = [customer() for customer in customers]
        assert sum(config.customer_mix) == 1, "The proportions for setting up the customer mix must sum to 1."
        return customers


    def simulate_customer_arrivals(self, simulation_mode):
        # linearly changing customer mix
        if config.linearly_changing_customers and not simulation_mode:
            config.customer_mix = [1 - self.step_counter / config.total_training_steps, self.step_counter / config.total_training_steps]
            if min(config.customer_mix) < 0:
                config.customer_mix = [0, 1]
        
        # usual drawing
        if config.stochastic_customers:
            return np.random.multinomial(config.n_customers, config.customer_mix)
        else:
            return np.multiply(config.customer_mix, config.n_customers)


    def reset(self, seed = 0):
        self.s = np.array([0, *[0 for _ in range(self.n_waiting_types + config.n_timesteps_saving)]])
        return self.s # , {}
