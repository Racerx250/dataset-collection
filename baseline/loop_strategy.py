import typing
import random

import torch

import ic_dataset

def get_subset(D: torch.utils.data.Dataset, ind:typing.Set[int]) -> torch.utils.data.Subset:
    return torch.utils.data.Subset(D, ind)

class FilterStrategy:
    def __init__(self):
        pass

    def filter(self, D):
        # raise Exception('FilterStrategy.filter not implemented!')
        pass

    def train(self, D_train):
        # raise Exception('FilterStrategy.train not implemented!')
        pass

class OracleStrategy:
    def __init__(self):
        pass

    def predict(self, images):
        # raise Exception('OracleStrategy.predict not implemented!')
        pass

class CombineStrategy:
    def __init__(self):
        pass

    def combine(self, D_0, D_1):
        # raise Exception('CombineStrategy.combine not implemented!')
        pass

class RandomFilter(FilterStrategy):
    perc = .1
    D = None

    def __init__(self, D, perc:float = .1):
        if perc > 1 or perc < 0: raise Exception('need 0 <= perc <= 1')
        self.perc = perc
        self.D = D

    def filter(self, D_test:typing.Set[int]) -> typing.Set[int]:
        return set([i for i in D_test if random.random() < self.perc])

    def train(self, D_train:typing.Set[int]) -> None:
        pass

class RandomOracle(OracleStrategy):
    perc = .9
    D = None

    def __init__(self, D:ic_dataset.ICDataset, perc:float = .9):
        if perc > 1 or perc < 0: raise Exception('need 0 <= perc <= 1')
        self.perc = perc
        self.D = D

    def predict(self, D_1_ind:typing.List[int]):
        label_map = [(i, int(self.D[i][1].numpy())) if random.random() < self.perc \
            else random.choice([(i, x) for x in range(len(self.D.all_labels)) if x != int(self.D[i][1].numpy())]) \
                for i in D_1_ind]

        return {l[0]:l[1] for l in label_map}

class SimpleCombine(CombineStrategy):
    D_0 = None

    def combine(self, D_0:dict, D_1:dict) -> dict:       
        temp = D_0.copy()
        temp.update(D_1)
        return temp

def start_loop(N:int, filtr:FilterStrategy, oracle:OracleStrategy, combiner:CombineStrategy, D_0_ind, D) -> None:
    if N < 1: raise Exception('N < 1')

    L_ind = set(range(len(D)))
    L_ind = L_indL.difference(D_0_ind)

    D_0 = {i:int(D[i][1].numpy()) for i in D_0_ind}

    for i in range(N):
        # train model if needed
        filtr.train(L_ind)
        
        # find new images
        D_1_ind = filtr.filter(L_ind)
        
        # predict labels using oracle
        D_1 = oracle.predict(D_1_ind)
        
        # combine the data
        D_0 = combiner.combine(D_0, D_1)

        L_ind = L_ind.difference(set(D_0.keys()))
    
    return None
    
if __name__ == '__main__':
    dataset = ic_dataset.get_icdataset('dataset_dogs_small_dirty')

    filtr = RandomFilter(dataset)
    oracle = RandomOracle(dataset)
    combiner = SimpleCombine()

    start_loop(10, filtr, oracle, combiner, set(range(10)), dataset)