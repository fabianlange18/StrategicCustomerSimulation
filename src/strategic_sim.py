from market.simulation import simulate_policy
from stable_baselines3.ppo import PPO
model = PPO.load('/Users/fabian/Developer/HPI/Masterarbeit/StrategicCustomerSimulation/results/02/02_Monopol_no_storage/model.zip')
infos = simulate_policy(model, True)

from evaluation.evaluator import Evaluator

ev = Evaluator()

ev.plot_trajectories(infos, save="RL_based_Monopol_fixed")
ev.print_simulation_statistics(infos)

# print(infos)