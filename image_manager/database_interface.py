import typing
import os
import pathlib
import traceback
import random

import numpy as np

from os import listdir
from os.path import isfile, join
from PIL import Image
from numpy import asarray
from typing import Union

PROJECT_DIR = str(pathlib.Path(__file__).parent.absolute())

class DatabaseInterface:
    db_name = None
    database_dict = None
    classes = []
    sources = []

    def __init__(self, db_name:str, database_dict:dict, sources:typing.List[str] = []):
        self.db_name = db_name
        self.database_dict = database_dict
        self.classes = list(database_dict.keys())
        self.sources = sources

    def get_db_name(self) -> str:
        return self.db_name

    def get_database_dict(self) -> dict:
        return self.database_dict

    def get_classes(self) -> typing.List[str]:
        return self.sources

    def get_source(self) -> typing.List[str]:
        return self.sources

    def set_class(self, from_class:str, num:int, to_class:str) -> dict:
        '''
        Returns the new num of the shifted image
        '''
        try:
            image_dict = self.database_dict[from_class][num]
        except Exception as e:
            print(f'{self.db_name}: {class_name}, {num} not found! (err code: 1)')
            return None

        del self.database_dict[from_class][num]

        new_num = max([elem.get('image_num', 0) for elem in self.database_dict[to_class]])
        self.database_dict[to_class][new_num] = image_dict

        return new_num

    def get_image(self, image:dict, raw:bool = False) -> Union[Image.Image, np.ndarray]:
        img = None
        try:
            if isinstance(image.get('full_path'), str): img = Image.open(image.get('full_path'))
            elif isinstance(image.get('relative_path'), str): img = Image.open(image.get('relative_path'))
            
            if not img: raise Exception('bad path on image: ' + str(image))
        except Exception as e:
            print(traceback.format_exc())
            return None

        if raw: return img

        return np.asarray(img)
        
    def get_images(self, images:typing.List[dict], raw:bool = False) -> Union[typing.List[Image.Image], typing.List[np.ndarray]]:
        np_images = []
        for image in images:
            img = self.get_image(image, raw=raw)
            if img is not None: np_images.append(img)

        return np_images

    def get_image_by_class_num(self, class_name:str, num:int, raw:bool = False) -> Union[typing.List[Image.Image], np.ndarray]:
        image_dict = None
        try:
            image_dict = self.database_dict[class_name][num]
        except Exception as e:
            print(f'{self.db_name}: {class_name}, {num} not found! (err code: 1)')
            return None

        if not image_dict or not isinstance(image_dict, dict):
            print(f'{self.db_name}: {class_name}, {num} not found! (err code: 2)')
            return None

        return self.get_image(image_dict, raw=raw)

    def get_images_by_class_num(self, images:typing.List[dict], raw:bool = False) -> Union[typing.List[Image.Image], typing.List[np.ndarray]]:
        np_images = []
        for image in images:
            if 'class' not in image or 'num' not in image: 
                print(f'{image} is not well formatted')
                continue
            
            img = get_image_by_class_num(image.get('class'), image.get('num'), raw=True)
            if img is not None: np_images.append(img)

        return np_images

    def copy(self):
        return DatabaseInterface()

class BlindDatabase:
    db_interface = None
    size = 0
    num_to_image_list = None

    def __init__(self, db_interface:DatabaseInterface):
        self.db_interface = db_interface

        db_dict = db_interface.get_database_dict()
        self.size = sum([len(db_dict[label]) for label in db_dict])

        self.num_to_image_list = [image_dict  for label in db_dict for image_dict in db_dict[label]]
        self.label_list = [label for label in db_dict for image_dict in db_dict[label]]

    def get_num_images(self) -> int:
        return len(self.num_to_image_list)

    def get_labels(self) -> typing.List:
        return list(set(self.oracle_labels(list(range(self.get_num_images())))))

    def oracle_label(self, num:int) -> str:
        if num > len(self.num_to_image_list) - 1: 
            raise Exception(f"{num} is larger than database size of {len(self.num_to_image_list)}! (remember 0 index)")
        
        return self.label_list[num]

    def oracle_labels(self, nums:typing.List[int]) -> typing.List:
        if not set(nums).issubset(set(range(self.size))): 
            raise Exception(f"Input index array is invalid!")
        
        return [self.label_list[num] for num in nums]

    def get_image_by_num(self, num:int, raw:bool = True) -> Union[Image.Image, np.ndarray]:
        if num > self.size - 1: 
            raise Exception(f"{num} is larger than database size of {len(self.num_to_image_list)}! (remember 0 index)")
        
        return self.db_interface.get_image(self.num_to_image_list[num], raw=True)

    def get_images_by_num(self, nums:typing.List[int], raw:bool) -> Union[Image.Image, typing.List[np.ndarray]]:
        if not set(nums).issubset(set(range(self.size))): 
            raise Exception(f"Input index array is invalid!")

        return self.db_interface.get_images([self.num_to_image_list[num] for num in nums], raw=True)

class DirtyDatabase(BlindDatabase):
    def __init__(self, db_interface:DatabaseInterface):
        super().__init__(db_interface)
        random.shuffle(self.label_list)

def get_database(db_name:str, dir_name:str = None, classes:typing.List[str] = [], sources:typing.List[str] = []) -> DatabaseInterface:
    path_to_dir = dir_name
    if not dir_name: path_to_dir = db_name

    class_dirs = [f for f in listdir(path_to_dir) if os.path.isdir(join(path_to_dir, f))]
    if len(classes) != 0: class_dirs = set(class_dirs).intersection(set(classes))
    
    all_sources = set({})
    class_image_map = {}
    for class_dir in class_dirs:
        joined_class_path = join(path_to_dir, class_dir)
        source_dirs = [f for f in listdir(joined_class_path) if os.path.isdir(join(joined_class_path, f))]
        
        if len(sources) != 0: source_dirs = list(set(source_dirs).intersection(set(sources)))
        all_sources.update(set(source_dirs))

        image_names = []
        image_num = 0
        for source_dir in [join(joined_class_path, source) for source in source_dirs]:
            image_names.extend([{
                'file_name': f,
                'dir': source_dir,
                'image_num': image_num,
                'relative_path': join(source_dir, f),
                'full_path': None
            } for f in listdir(source_dir) if isfile(join(source_dir, f)) and f[-5:] != '.json'])
            image_num += 1

        class_image_map[class_dir] = image_names

    return DatabaseInterface(db_name, class_image_map, list(all_sources))

def create_blind_database(db_name:str, dir_name:str = None, classes:typing.List[str] = [], sources:typing.List[str] = []) -> DirtyDatabase:
    database_interface = get_database(db_name, dir_name=dir_name, classes=classes, sources=sources)
    return BlindDatabase(database_interface)

def create_dirty_database(db_name:str, dir_name:str = None, classes:typing.List[str] = [], sources:typing.List[str] = []) -> DirtyDatabase:
    database_interface = get_database(db_name, dir_name=dir_name, classes=classes, sources=sources)
    return DirtyDatabase(database_interface)

def split_db_dict(db_name:str, dir_name:str = None, classes:typing.List[str] = [], sources:typing.List[str] = [], train_perc:float = 1) -> dict:
    if train_perc < 0 or train_perc > 1: raise Exception('need 0 <= train_perc <= 1')

    database_interface = get_database(db_name, dir_name=dir_name, classes=classes, sources=sources)
    
    train_dict = {}
    test_dict = {}

    db_dict = database_interface.get_database_dict()
    for label in db_dict:
        train_dict[label] = db_dict[label][0:int(len(db_dict[label])*train_perc)]
        test_dict[label] = db_dict[label][int(len(db_dict[label])*train_perc):]

    return BlindDatabase(DatabaseInterface(db_name + '_train', train_dict, sources=sources)), BlindDatabase(DatabaseInterface(db_name + '_test', test_dict, sources=sources))

if __name__ == '__main__':
    # NEED TO MOVE BELOW TO TESTS
    pass

    # temp = get_database('dataset_dogs_small_dirty')
    # print(temp.get_image_by_class_num('border_collie_dog', 0, raw=True))

    # temp = create_blind_database('dataset_dogs_small_dirty')
    # print(temp.oracle_labels(list(range(10))))
    # print(temp.get_image_by_num(5, raw=True))
    # print(temp.get_labels())

    # print(split_db_dict('dataset_dogs_small_dirty', train_perc=.7))
