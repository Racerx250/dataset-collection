import os
from os import listdir
from os.path import isfile, join
import json
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import seaborn as sns

from PIL import Image, ImageStat

def clean_dataset_dir(dir_name:str):
    path_to_dir = dir_name
    removed_num = 0

    class_dirs = [f for f in listdir(path_to_dir) if os.path.isdir(join(path_to_dir, f))]
    
    final_counts = {}
    dim_map = {}
    total_num = 0
    it_num = 0
    overal_dims = {}
    final_greyscales = {}
    final_sizes = {}
    class_final_sizes = {}
    for class_dir in class_dirs:
        joined_class_path = join(path_to_dir, class_dir)
        source_dirs = [f for f in listdir(joined_class_path) if os.path.isdir(join(joined_class_path, f))]
        
        class_dim_map = {}
        greyscale_num = 0
        greyscale_dirs = []
        class_sizes = {}
        for source_dir in [join(joined_class_path, source) for source in source_dirs]:
            label_num = len([f for f in listdir(source_dir) if isfile(join(source_dir, f)) and f[-5:] != '.json'])
            final_counts[class_dir] = label_num
            total_num += label_num

            temp = [Image.open(join(source_dir, f)).size for f in listdir(source_dir) if isfile(join(source_dir, f)) and f[-5:] != '.json']
            div_sizes = [str(round(f[1]/f[0], 2)) for f in temp]
            for dim in div_sizes:
                if dim in dim_map: dim_map[dim] += 1
                else: dim_map[dim] = 1
                
                if dim in class_dim_map: class_dim_map[dim] += 1
                else: class_dim_map[dim] = 1

            for im in temp:
                trunc_size = f"{int(im[0]/100)*100}, {int(im[1]/100)*100}"

                if trunc_size in final_sizes: final_sizes[trunc_size] += 1
                else: final_sizes[trunc_size] = 1

                if trunc_size in class_sizes: class_sizes[trunc_size] += 1
                else: class_sizes[trunc_size] = 1

            # for f in listdir(source_dir):
            #     if isfile(join(source_dir, f)) and f[-5:] != '.json': 
            #         im = Image.open(join(source_dir, f)).convert("RGB")
            #         stat = ImageStat.Stat(im)
                    
            #         if sum(stat.sum)/3 == stat.sum[0]: 
            #             greyscale_dirs.append(join(source_dir, f))
            #             greyscale_num += 1
        
        final_greyscales[class_dir] = {}
        final_greyscales[class_dir]['dirs'] = greyscale_dirs
        final_greyscales[class_dir]['count'] = greyscale_num          
        overal_dims[class_dir] = class_dim_map
        class_final_sizes[class_dir] = class_sizes

    final_counts['total'] = total_num

    with open('class_counts.json', 'w') as f: json.dump(final_counts, f, indent=2)
    with open('dim_counts.json', 'w') as f: json.dump(dim_map, f, indent=2)
    with open('overall_dims.json', 'w') as f: json.dump(overal_dims, f, indent=2)
    # with open('greyscale.json', 'w') as f: json.dump(final_greyscales, f, indent=2)
    with open('sizes.json', 'w') as f: json.dump(final_sizes, f, indent=2)
    with open('class_sizes.json', 'w') as f: json.dump(class_final_sizes, f, indent=2)

def create_dim_table():
    test = None
    with open('sizes.json', 'r') as f: test = json.load(f)

    df = pd.DataFrame(list(zip(test[label].keys(),test[label].values())), columns=['dim','count'])
    temp = df.sort_values(by=['dim'])
    plt.figure(figsize=(20,5))
    plt.bar(temp['dim'].tolist(), temp['count'].tolist(),.5, color='g')
    plt.savefig(f"dataset_test_images/by_class/dim_counts_{label}.jpg")
        

def create_graphics():
    # cur_time = datetime.datetime.now()
    # dataset_name = 'dataset_' + 'metrics' + '_' + cur_time.strftime("%m-%d-%YT%H:%M:%S")
    # if not os.path.exists(dataset_name):
    #     os.makedirs(dataset_name)

    test = None
    # with open('class_counts.json', 'r') as f: test = json.load(f)
    # with open('dim_counts.json', 'r') as f: test = json.load(f)
    with open('overall_dims.json', 'r') as f: test = json.load(f)
    # df = pd.DataFrame(list(zip(test.keys(),test.values())), columns=['dim','count'])
    # test = df.sort_values(by=['dim'])

    for label in test.keys():
        df = pd.DataFrame(list(zip(test[label].keys(),test[label].values())), columns=['dim','count'])
        temp = df.sort_values(by=['dim'])
        plt.figure(figsize=(20,5))
        plt.bar(temp['dim'].tolist(), temp['count'].tolist(),.5, color='g')
        plt.savefig(f"dataset_test_images/by_class/dim_counts_{label}.jpg")
        plt.clf()
        # break
        
    # print(test)
    # print(test['dim'].tolist())
    # print(df.sort_values(by=['dim']))

    # plt.figure(figsize=(100,5))
    # # plt.bar(list(test.keys()), test.values(), .5, color='g')
    # plt.bar(test['dim'].tolist(), test['count'].tolist(),.5, color='g')
    # plt.savefig(f"dataset_test_images/dim_counts_2.jpg")

if __name__ == '__main__':
    # create_graphics()
    # create_dim_table()
    clean_dataset_dir('dataset_stanford_dog_recreation')
    # pass