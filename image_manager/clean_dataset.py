import os
from os import listdir
from os.path import isfile, join

from PIL import Image

def clean_dataset_dir(dir_name:str):
    path_to_dir = dir_name
    removed_num = 0

    class_dirs = [f for f in listdir(path_to_dir) if os.path.isdir(join(path_to_dir, f))]
    for class_dir in class_dirs:
        joined_class_path = join(path_to_dir, class_dir)
        source_dirs = [f for f in listdir(joined_class_path) if os.path.isdir(join(joined_class_path, f))]
        
        for source_dir in [join(joined_class_path, source) for source in source_dirs]:
            for f in listdir(source_dir):
                if isfile(join(source_dir, f)) and f[-5:] != '.json':
                    try:
                        Image.open(join(source_dir, f))
                    except:
                        os.remove(join(source_dir, f))
                        removed_num += 1
    
    print(f"Removed {removed_num} images.")

if __name__ == '__main__':
    # clean_dataset_dir('dataset_stanford_dog_recreation')
    pass