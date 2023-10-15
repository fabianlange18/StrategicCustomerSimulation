import config
import numpy as np

def softmax(x):
    probabilities = np.exp(x) / np.sum(np.exp(x), axis=0)
    if config.edgeworth:
        max_index = list(probabilities).index(max(probabilities))
        result = [0] * len(probabilities)
        result[max_index] = 1
        return np.array(result)
    else:
        return probabilities