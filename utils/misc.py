import numpy as np
import json

def get_random_sents(sents, n=10, random_seed=None):
    if isinstance(random_seed, int):
        np.random.seed(random_seed)
    return np.random.choice(sents, n)

def load(file_name):
    with open(file_name, "r") as f:
        d = json.load(f)
        return d

def store(file_name, contents):
    obj = json.dumps(contents, indent=2)
    with open(file_name, "w") as f:
        f.write(obj)