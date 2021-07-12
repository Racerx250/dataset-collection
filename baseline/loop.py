import typing
import random
import json
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from collections import defaultdict
import pickle
import os.path
import math
import sys

import train_custom
import ic_dataset

def get_subset(D: torch.utils.data.Dataset, ind:typing.Set[int]) -> torch.utils.data.Subset:
    replaced_dataset = [(D[i][0].numpy(), np.array(ind[i]) if i in ind.keys() else D[i][1].numpy()) for i in range(len(D))]
    return torch.utils.data.Subset(replaced_dataset, list(ind))

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
    test_set = None

    def __init__(self, D, test_set, perc:float = .1):
        if perc > 1 or perc < 0: raise Exception('need 0 <= perc <= 1')
        self.perc = perc
        self.D = D
        self.test_set = test_set

    def filter(self, D_test:typing.Set[int]) -> typing.Set[int]:
        return set([i for i in D_test if random.random() < self.perc])

class NNFilter(FilterStrategy):
    perc = .1
    D = None
    test_set = None
    loopNum = None

    def __init__(self, D, test_set, loopNum, perc:float = .1):
        if perc > 1 or perc < 0: raise Exception('need 0 <= perc <= 1')
        self.perc = perc
        self.D = D
        self.test_set = test_set
        self.loopNum = loopNum

    def filter(self, D_test:typing.Set[int]) -> typing.Set[int]:
        return set([i for i in D_test if random.random() < self.perc])

    def train(self, sizeNum, D_train:dict) -> None:
        total_data = get_subset(self.D, D_train)
        train_len = round(len(total_data)*0.85)
        train_set, val_set = torch.utils.data.random_split(total_data, [train_len, len(total_data)-train_len])

        torch.cuda.empty_cache()
        model = models.inception_v3(pretrained=True, init_weights=True, aux_logits=True)
        model.fc = nn.Linear(2048, 120)
        model.AuxLogits.fc = nn.Linear(768, 120)
    
        '''
        train_transform = transforms.Compose([ 
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        test_transform = transforms.Compose([ 
            transforms.Resize(299),
            transforms.CenterCrop(299), 
            transforms.ToTensor()])

       
        train_set.dataset.transform = train_transform
        print(train_set.dataset.transform)
        val_set.dataset.transform = test_transform
        print(val_set.dataset.transform)
        test_set.transform = test_transform
        print(test_set.dataset.transform)
        '''

        with open("config.json", "r") as jsonfile:
            configData = json.load(jsonfile)
        train_loader = DataLoader(train_set, shuffle=True, batch_size = configData['batch_size'], num_workers=configData['num_workers'], drop_last = True)
        val_loader =  DataLoader(val_set, shuffle=False, num_workers=4)
        test_loader = DataLoader(self.test_set, shuffle=False, num_workers=4)

        optimizer = torch.optim.Adam(model.parameters(), lr=configData['lr'], weight_decay=configData['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        model = model.to('cuda')

        train_custom.train_model(model, train_loader, val_loader, test_loader, configData['epoch'], 
                optimizer, criterion, configData['patience'], True, self.loopNum, sizeNum, configData['graphToggle'])
        del train_set
        del val_set
        del train_loader
        del val_loader
        del test_loader

class RandomOracle(OracleStrategy):
    perc = .9
    D = None

    def __init__(self, D:ic_dataset.ICDataset, perc:float = .9):
        if perc > 1 or perc < 0: raise Exception('need 0 <= perc <= 1')
        self.perc = perc
        self.D = D

    def predict(self, D_1_ind:typing.List[int]) -> dict:
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
    L_ind = L_ind.difference(D_0_ind)

    D_0 = {i:int(D[i][1].numpy()) for i in D_0_ind}

    for i in range(N):
        # train model if needed
        filtr.train(i, D_0)
        
        # find new images
        D_1_ind = filtr.filter(L_ind)
        
        # predict labels using oracle
        D_1 = oracle.predict(D_1_ind)
        
        # combine the data
        D_0 = combiner.combine(D_0, D_1)

        L_ind = L_ind.difference(set(D_0.keys()))

    with open('D_0_final.json', 'w') as f: json.dump(D_0, f, indent=2)
    
if __name__ == '__main__':
    # print(ic_dataset.get_icdataset('dataset_dogs_large'))
    # dataset, test_set = ic_dataset.get_icdataset_train_test('dataset_dogs_large', train_perc=0.85)
    if (os.path.isfile('dataset.pkl')) :
        with open('dataset.pkl', 'rb') as input:
            dataset = pickle.load(input)
        with open('test_set.pkl', 'rb') as input:
            test_set = pickle.load(input)
    else :
        with open("config.json", "r") as jsonfile:
            configData = json.load(jsonfile)
        dataset, test_set = ic_dataset.get_icdataset_train_test(configData['dataPath'], train_perc=0.85)
        with open('dataset.pkl', 'wb') as output:
            pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)
        with open('test_set.pkl', 'wb') as output:
            pickle.dump(test_set, output, pickle.HIGHEST_PROTOCOL)

    #print("finish importing images")
    start_perc = .1
    D_0_ind = set(random.sample(range(len(dataset)), int(len(dataset)*start_perc)))
    # get accuracy from command arg
    accuracy = int(sys.argv[1])
    oracle = RandomOracle(dataset, accuracy/10)
    filtr = NNFilter(dataset, test_set, accuracy, perc=.1)
    combiner = SimpleCombine()
    #print("start loop")
    start_loop(10, filtr, oracle, combiner, D_0_ind, dataset)

        
