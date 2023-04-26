import numpy as np
from sentence_transformers.losses import InputExample

def generate_triples(split_data, dropout=0.9):
    n = len(split_data)
    triples = []
    for i in range(n):
        sent, label = split_data[i]["sentence"], split_data[i]["label"]
        sarc, non_sarc = split_data[i]["sarc"], split_data[i]["non_sarc"]
        
        # case distinction based on whether sent is sarcastic or not
        [iter1, iter2] = [sarc, non_sarc] if label else [non_sarc, sarc]
        for s1 in iter1:
            for s2 in iter2:
                if np.random.rand() < dropout: continue
                triples.append((sent, s1, s2))
                    
    return triples

def generate_tuples(split_data, dropout=0.2):
    n = len(split_data)
    tuples = []
    for i in range(n):
        sent, label = split_data[i]["sentence"], split_data[i]["label"]
        sarc, non_sarc = split_data[i]["sarc"], split_data[i]["non_sarc"]
        
        iter = sarc if label else non_sarc
        for s in iter:
            if np.random.rand() < dropout: continue
            tuples.append((sent, s))
                    
    return tuples

def generate_labelled_tuples(split_data, dropout=0.2):
    n = len(split_data)
    labelled_tuples = []
    for i in range(n):
        sent, label = split_data[i]["sentence"], split_data[i]["label"]
        sarc, non_sarc = split_data[i]["sarc"], split_data[i]["non_sarc"]
        
        [maximize, minimize] = [non_sarc, sarc] if label else [sarc, non_sarc]
        
        for target, data in enumerate([maximize, minimize]):
            for s in data:
                if np.random.rand() < dropout: continue
                labelled_tuples.append(((sent, s), target))
                    
    return labelled_tuples
        

# Example generation as described in the paper
def generate_training_examples(split_data, model_name, dropout=0, random_seed=None):
    if isinstance(random_seed, int):
        np.random.seed(random_seed)

    examples = []
    if "Contrastive" in model_name:
        examples = generate_labelled_tuples(split_data, dropout=dropout)
        return [InputExample(texts=texts, label=label) for texts, label in examples]
    else:
        examples = []
        if "Triplet" in model_name:
            examples = generate_triples(split_data, dropout=dropout) 
        elif "MultipleNegatives" in model_name:
            examples = generate_tuples(split_data, dropout=dropout)
    
        return [InputExample(texts=example) for example in examples]
    