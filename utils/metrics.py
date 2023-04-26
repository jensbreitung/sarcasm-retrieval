import numpy as np
from sentence_transformers.util import cos_sim
import torch

def load_weights(k: int, use_discounting=False):
    w = np.ones(k) / k
    
    if use_discounting:
        # linear discounting
        w = np.array([k-i for i in range(k)])
        w = w / np.sum(w)
    
    return w

# implementation of polarity and semantic similarity score as in the paper (use_discounting=True)

def polarity_score(y, ys, use_discounting=False):
    k = len(ys)

    # setup weights
    w = load_weights(k, use_discounting)
    
    return np.sum(w[i] * (1 - y^ys[i]) for i in range(k))
    
def semantic_similarity_score(sentence, suggestions, reference_model, use_discounting=False):
    k = len(suggestions)
        
    # embed sentence and suggestions and store as torch tensors
    emb = torch.tensor(reference_model.embed(sentence))
    embs = [torch.tensor(vec) for vec in reference_model.embed(suggestions)]

    # setup weights
    w = load_weights(k, use_discounting)

    # compute weighted average of cos_sim values
    similarities = [cos_sim(emb, embs[i]).item() for i in range(k)]
    return np.sum(w[i] * similarities[i] for i in range(k))