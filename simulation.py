import config
from market.market import Market

from tqdm import trange

import numpy as np

def simulate_policy(model):

    e = Market()
    s_next = e.s

    infos = {}

    print(f"\nSimulating Environment")
    for i in trange(config.number_of_simulation_episodes * config.episode_length):

        action = model.predict(s_next, deterministic=True)[0]
        s_next, reward, done, info = e.step(np.array(action), simulation_mode=True)
    
        for key in info.keys():
            if i == 0:
                infos[key] = []
            infos[key].append(info[key])

        if done:
            e.reset()
    
    return infos