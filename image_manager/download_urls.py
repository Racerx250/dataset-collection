import requests
import csv 
import urllib.request
import time
import datetime
import itertools
import concurrent
import os
import socket

from concurrent.futures import ThreadPoolExecutor

socket.setdefaulttimeout(10)

def get_row(filename):
    for line in open(filename, "rb"):
        yield str(line).split(',')

def create_new_dir():
    cur_time = datetime.datetime.now()
    dataset_name = 'dataset_' + 'dogs' + '_' + cur_time.strftime("%m-%d-%YT%H:%M:%S")

    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    return dataset_name, get_label_map(dataset_name)

def get_label_map(dataset_name):
    label_map = {}
    classes_generator = get_row('../data/dog_classes.txt')
    for label in classes_generator:
        trim_label = label[0].replace("b\'", "").replace('\"', '')
        trim_id = label[1].replace("\\r\\n", "").replace('\'', '')
        if not os.path.exists(os.path.join(dataset_name, f'{trim_label}')):
            os.makedirs(os.path.join(dataset_name, f'{trim_label}'))
        label_map[trim_id] = trim_label

    return label_map

def save_image(img_tuple, dataset_name, image_num, completed_set):
    trim_id = img_tuple[0].replace("b\'", "").replace("\\r\\n", "").replace('\'', '')

    if image_num < completed_set: return -1
    # if ','.join(img_tuple[:2]).replace("b\'", "").replace("\\n\'", "") in completed_set: return -1

    errors = []
    for img_url in img_tuple[1:]:
        trim_url = img_url.replace("'", "").replace('\\n', '')
        try:
            urllib.request.urlretrieve(trim_url, os.path.join(dataset_name, label_map[trim_id], f'{image_num}.jpg'))
            return True, None, img_tuple
        except Exception as e:                
            errors.append(f'{image_num}, {e}, {trim_url}')
            pass
    return False, errors, img_tuple

if __name__ == '__main__':
    img_generator = get_row('../data/dog_urls.txt')

    # dataset_name, label_map = create_new_dir()
    dataset_name = 'dataset_dogs_04-29-2021T12:27:02'
    label_map = get_label_map(dataset_name)

    # completed_set = []
    # if os.path.isfile(os.path.join(dataset_name, 'completed.txt')):
    #     completed_generator = list(get_row(os.path.join(dataset_name, 'completed.txt')))
    #     completed_set = [','.join(c[:2]).replace("b\'", "").replace("\\n\'", "") for c in completed_generator]
    # completed_set = len(completed_set)
    completed_set = 43911

    num_workers = 100
    requested_num = num_workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(save_image, img_tuple, dataset_name, i, completed_set)
            for i, img_tuple in enumerate(itertools.islice(img_generator, num_workers))
        }
        
        while futures:
            done, futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for d in done:
                res = d.result()
                if isinstance(res, int): continue
                if not res[0]: 
                    with open(os.path.join(dataset_name, 'errors.txt'), 'a') as f:
                        for s in res[1]: f.write(s + '\n')
                with open(os.path.join(dataset_name, 'completed.txt'), 'a') as f:
                    f.write((','.join(res[2][:2])).replace("b\'", "").replace("\\n\'", "") + '\n')

            for i, img_tuple in enumerate(itertools.islice(img_generator, len(done))):
                futures.add(
                    executor.submit(save_image, img_tuple, dataset_name, i + requested_num, completed_set)
                ) 
                requested_num += 1

                if requested_num % 100 == 0:
                    print(requested_num)
            
            # time.sleep(1)
        
        