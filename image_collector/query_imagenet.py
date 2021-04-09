#!/usr/bin/env python3
import os
import numpy as np
import requests
import argparse
import json
import time
import logging
import csv
from multiprocessing import Pool, Process, Value, Lock
from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL

# ALL (well most, had to clean up a lot) CREDIT GOES TO:
# https://github.com/mf1024/ImageNet-Datasets-Downloader

IMAGENET_API_WNID_TO_URLS = lambda wnid: f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={wnid}'
current_folder = os.path.dirname(os.path.realpath(__file__))

class_info_json_filename = 'imagenet_class_info.json'
class_info_json_filepath = os.path.join(current_folder, class_info_json_filename)
class_info_dict = dict()

with open(class_info_json_filepath) as class_info_json_f:
    class_info_dict = json.load(class_info_json_f)


# 
# 
# 


scraping_stats = dict(
    all=dict(
        tried=0,
        success=0,
        time_spent=0,
    ),
    is_flickr=dict(
        tried=0,
        success=0,
        time_spent=0,
    ),
    not_flickr=dict(
        tried=0,
        success=0,
        time_spent=0,
    )
)

def add_debug_csv_row(row):
    with open('stats.csv', "a") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",")
        csv_writer.writerow(row)

class MultiStats():
    def __init__(self):

        self.lock = Lock()

        self.stats = dict(
            all=dict(
                tried=Value('d', 0),
                success=Value('d',0),
                time_spent=Value('d',0),
            ),
            is_flickr=dict(
                tried=Value('d', 0),
                success=Value('d',0),
                time_spent=Value('d',0),
            ),
            not_flickr=dict(
                tried=Value('d', 0),
                success=Value('d', 0),
                time_spent=Value('d', 0),
            )
        )
    def inc(self, cls, stat, val):
        with self.lock:
            self.stats[cls][stat].value += val

    def get(self, cls, stat):
        with self.lock:
            ret = self.stats[cls][stat].value
        return ret

multi_stats = MultiStats()
row = [
    "all_tried",
    "all_success",
    "all_time_spent",
    "is_flickr_tried",
    "is_flickr_success",
    "is_flickr_time_spent",
    "not_flickr_tried",
    "not_flickr_success",
    "not_flickr_time_spent"
]
add_debug_csv_row(row)

def add_stats_to_debug_csv():
    row = [
        multi_stats.get('all', 'tried'),
        multi_stats.get('all', 'success'),
        multi_stats.get('all', 'time_spent'),
        multi_stats.get('is_flickr', 'tried'),
        multi_stats.get('is_flickr', 'success'),
        multi_stats.get('is_flickr', 'time_spent'),
        multi_stats.get('not_flickr', 'tried'),
        multi_stats.get('not_flickr', 'success'),
        multi_stats.get('not_flickr', 'time_spent'),
    ]
    add_debug_csv_row(row)

def print_stats(cls, print_func):

    actual_all_time_spent = time.time() - scraping_t_start.value
    processes_all_time_spent = multi_stats.get('all', 'time_spent')

    if processes_all_time_spent == 0:
        actual_processes_ratio = 1.0
    else:
        actual_processes_ratio = actual_all_time_spent / processes_all_time_spent

    #print(f"actual all time: {actual_all_time_spent} proc all time {processes_all_time_spent}")

    print_func(f'STATS For class {cls}:')
    print_func(f' tried {multi_stats.get(cls, "tried")} urls with'
               f' {multi_stats.get(cls, "success")} successes')

    if multi_stats.get(cls, "tried") > 0:
        print_func(f'{100.0 * multi_stats.get(cls, "success")/multi_stats.get(cls, "tried")}% success rate for {cls} urls ')
    if multi_stats.get(cls, "success") > 0:
        print_func(f'{multi_stats.get(cls,"time_spent") * actual_processes_ratio / multi_stats.get(cls,"success")} seconds spent per {cls} succesful image download')

lock = Lock()
url_tries = Value('d', 0)
scraping_t_start = Value('d', time.time())
class_folder = ''
class_images = Value('d', 0)

def get_image(img_url, images_per_class):

    if len(img_url) <= 1:
        return

    cls_imgs = 0
    with lock:
        cls_imgs = class_images.value

    if cls_imgs >= images_per_class:
        return

    logging.debug(img_url)

    cls = ''

    if 'flickr' in img_url:
        cls = 'is_flickr'
    else:
        cls = 'not_flickr'

    t_start = time.time()

    def finish(status):
        t_spent = time.time() - t_start
        multi_stats.inc(cls, 'time_spent', t_spent)
        multi_stats.inc('all', 'time_spent', t_spent)

        multi_stats.inc(cls,'tried', 1)
        multi_stats.inc('all', 'tried', 1)

        if status == 'success':
            multi_stats.inc(cls,'success', 1)
            multi_stats.inc('all', 'success', 1)

        elif status == 'failure':
            pass
        else:
            logging.error(f'No such status {status}!!')
            exit()
        return


    with lock:
        url_tries.value += 1
        if url_tries.value % 250 == 0:
            print(f'\nScraping stats:')
            print_stats('is_flickr', print)
            print_stats('not_flickr', print)
            print_stats('all', print)
            if args.debug:
                add_stats_to_debug_csv()

    try:
        img_resp = requests.get(img_url, timeout = 1)
    except ConnectionError:
        logging.debug(f"Connection Error for url {img_url}")
        return finish('failure')
    except ReadTimeout:
        logging.debug(f"Read Timeout for url {img_url}")
        return finish('failure')
    except TooManyRedirects:
        logging.debug(f"Too many redirects {img_url}")
        return finish('failure')
    except MissingSchema:
        return finish('failure')
    except InvalidURL:
        return finish('failure')
    except Exception as e:
        logging.debug(f"Invalid Schema? {img_url}")
        print(img_url)
        return
    if not 'content-type' in img_resp.headers:
        return finish('failure')

    if not 'image' in img_resp.headers['content-type']:
        logging.debug("Not an image")
        return finish('failure')

    if (len(img_resp.content) < 1000):
        return finish('failure')

    logging.debug(img_resp.headers['content-type'])
    logging.debug(f'image size {len(img_resp.content)}')

    img_name = img_url.split('/')[-1]
    img_name = img_name.split("?")[0]

    if (len(img_name) <= 1):
        return finish('failure')

    img_file_path = os.path.join(class_folder, img_name)
    logging.debug(f'Saving image in {img_file_path}')

    with open(img_file_path, 'wb') as img_f:
        img_f.write(img_resp.content)

        with lock:
            class_images.value += 1

        logging.debug(f'Scraping stats')
        print_stats('is_flickr', logging.debug)
        print_stats('not_flickr', logging.debug)
        print_stats('all', logging.debug)

        return finish('success')

def start_query(scrape_only_flickr=True, images_per_class=10, data_root='', class_list=None):
    classes_to_scrape = []
    # print(class_info_dict)

    if class_list and isinstance(class_list, list):
        for item in class_list:
            if item not in class_info_dict:
                logging.error(f'Class {item} not found in ImageNet')
            else:
                classes_to_scrape.append(item)

    imagenet_images_folder = os.path.join(data_root, 'imagenet_images')
    if not os.path.isdir(imagenet_images_folder):
        os.mkdir(imagenet_images_folder)

    query_images(imagenet_images_folder, classes_to_scrape, images_per_class)
    
def query_images(imagenet_images_folder, classes_to_scrape, images_per_class, num_workers=8):
    for class_wnid in classes_to_scrape:
        class_name = class_info_dict[class_wnid]["class_name"]
        print(f'Scraping images for class \"{class_name}\"')
        url_urls = IMAGENET_API_WNID_TO_URLS(class_wnid)

        time.sleep(0.05)
        resp = requests.get(url_urls)

        class_folder = os.path.join(imagenet_images_folder, class_name)
        if not os.path.exists(class_folder):
            os.mkdir(class_folder)

        class_images.value = 0

        urls = [url.decode('utf-8') for url in resp.content.splitlines()]
        print(urls)
        for url in resp.content.splitlines():
            get_image(url.decode('utf-8'), images_per_class)
            
        # print(f"Multiprocessing workers: {num_workers}")
        # with Pool(processes=num_workers) as p:
        #     p.map(get_image,urls)

start_query(class_list=['n02106166'])