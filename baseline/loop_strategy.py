import typing
import random

import torch

import ic_dataset

class FilterStrategy:
    def __init__(self):
        pass

    def filter(self, D)
        raise Exception('FilterStrategy.filter not implemented!')

    def train(self, D_train):
        raise Exception('FilterStrategy.train not implemented!')

class OracleStrategy:
    def __init__(self):
        pass

    def predict(self, images):
        raise Exception('OracleStrategy.predict not implemented!')

class CombineStrategy:
    def __init__(self):
        pass

    def combine(self, D_0, D_1):
        raise Exception('CombineStrategy.combine not implemented!')

class RandomFilter(FilterStrategy):
    perc = .1

    def __init__(self, perc:float = .1, ):
        if perc > 1 or perc < 0: raise Exception('need 0 <= perc <= 1')
        self.perc = perc

    def filter(self, D:ic_dataset.ICDataset):
        return torch.utils.data.Subset(D, [i for i in range(len(D)) if random.random() < perc])

    def train(self, D_train):
        pass

class RandomOracle(OracleStrategy):
    perc = .9
    D = None

    def __init__(self, perc:float = .9, D:ic_dataset.ICDataset):
        if perc > 1 or perc < 0: raise Exception('need 0 <= perc <= 1')
        self.perc = perc
        self.D = D

    def predict(self, images:typing.List[int]):
        return [int(D[i][1]) for i in images]

class SimpleCombine(CombineStrategy):
    D_0 = None

    def combine(self, D_0, D_1):
        self.D_0 = torch.utils.data.ConcatDataset([D_0, D_1])
        return self.D_0

def start_loop(N:int, filtr:FilterStrategy, oracle:OracleStrategy, combiner:CombineStrategy, D_0, D) -> None:
    if N < 1: raise Exception('N < 1')

    L = D.copy()
    # SET L
    
    for i in range(N):
        filtr.train(D_0)
        D_1 = filtr.filter(L)
        D_1 = oracle.predict(D_1)
        D_0 = combiner.combine(D_0, D_1)
        # SET L
    
if __name__ == '__main__':
    dataset = ic_dataset.get_icdataset('dataset_dogs_small_dirty')

    filtr = RandomFilter()
    oracle = RandomOracle(dataset)
    combiner = SimpleCombine()

    start_loop(10, filtr, oracle, combiner)