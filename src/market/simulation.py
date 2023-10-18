import config
from .market import Market
from customers._5_strategic.baseline import StrategicCustomer

import numpy as np
from tqdm import trange

def simulate_policy(model, deterministic, prog_bar=True):

    e = StrategicCustomer() if config.train_strategic else Market()
    s_next = e.s

    infos = {}

    if prog_bar:
        print(f"\nSimulating Environment")
        loop_range = trange(config.n_simulation_episodes * config.episode_length)
    else:
        loop_range = range(config.n_simulation_episodes * config.episode_length)
    
    for i in loop_range:

        action = model.predict(s_next, deterministic=deterministic)[0]
        s_next, reward, done, info = e.step(np.array(action), simulation_mode=True) # _ ,
        # s_next, reward, done, info = e.step(np.array([action]), simulation_mode=True) # _ ,
    
        for key in info.keys():
            if i == 0:
                infos[key] = []
            infos[key].append(info[key])

        if done:
            e.reset()
            if config.train_strategic:
                break
    
    return infos
