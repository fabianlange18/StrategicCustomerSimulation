import config
from market.market import Market

from tqdm import trange

import numpy as np

def simulate_policy(model, prog_bar=True):

    e = Market()
    s_next = e.s

    infos = {}

    if prog_bar:
        print(f"\nSimulating Environment")

    loop_range = trange(config.n_simulation_episodes * config.episode_length) if prog_bar else range(config.n_simulation_episodes * config.episode_length)
    
    for i in loop_range:

        action = model.predict(s_next, deterministic=True)[0]
        s_next, reward, done, info = e.step(np.array(action), simulation_mode=True)
    
        for key in info.keys():
            if i == 0:
                infos[key] = []
            infos[key].append(info[key])

        if done:
            e.reset()
    
    return infos