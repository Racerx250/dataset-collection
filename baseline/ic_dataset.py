import pathlib
import sys
import os
import json

PROJECT_DIR = str(pathlib.Path(__file__).parent.absolute().parent.absolute())
sys.path.insert(0, os.path.join(PROJECT_DIR, 'image_manager'))

import database_interface
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms as tv

# IC = Image Collector :)
class ICDataset(Dataset):
    def __init__(self, blind_database:database_interface.BlindDatabase, database_dir_path:str = None, db_name:str = None, use_int_labels:bool = False):
        self.normalize = tv.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = tv.Compose([tv.Resize(256), tv.CenterCrop(224), tv.ToTensor()])
        
        self.database_dir_path = database_dir_path
        self.use_int_labels = use_int_labels
        
        if not db_name: self.db_name = db_name
        else: self.db_name = database_dir_path

        self.blind_database = blind_database

        self.use_int_labels = use_int_labels
        self.all_labels = self.blind_database.get_labels()
        self.label_num_map = None

        if use_int_labels: self.label_num_map = {self.all_labels[i]:i for i in range(len(self.all_labels))}

    def save_label_map(self):
        if not self.label_num_map: return
        with open('label_map.json', 'w') as f: json.dump(self.label_num_map, f, indent=2)

    def __len__(self):
        return self.blind_database.get_num_images()

    def __getitem__(self, item):
        image = self.normalize(self.transform(self.blind_database.get_image_by_num(item).convert('RGB')))
        label = self.blind_database.oracle_label(item)
        if self.use_int_labels: label = self.label_num_map[label]

        return image, torch.from_numpy(np.asarray(label))

def get_icdataset(database_dir_path:str, db_name:str = None, use_int_labels:bool = True) -> ICDataset:
    return ICDataset(database_interface.create_blind_database(db_name or database_dir_path, dir_name=database_dir_path), use_int_labels=use_int_labels)

def get_icdataset_train_test(database_dir_path:str, db_name:str = '', train_perc:float = 1, use_int_labels:bool = True) -> ICDataset:
    if train_perc < 0 or train_perc > 1: raise Exception('need 0 <= train_perc <= 1')

    test_db, train_db = database_interface.split_db_dict(db_name or database_dir_path, dir_name=database_dir_path, train_perc=train_perc)

    return ICDataset(train_db, database_dir_path=database_dir_path, db_name=db_name, use_int_labels=use_int_labels), ICDataset(test_db, database_dir_path=database_dir_path, db_name=db_name, use_int_labels=use_int_labels)

if __name__ == '__main__':
    # NEED TO MOVE BELOW TO TESTS
    pass

    # temp = get_icdataset('dataset_dogs_small_dirty')
    # temp.save_label_map()
    # print(temp[0])

    # print(get_icdataset_train_test('dataset_dogs_small_dirty', train_perc=.7))
    


