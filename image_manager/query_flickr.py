from __future__ import print_function
import time
import sys
import json
import re
import os
import requests
import datetime
import pathlib
from tqdm import tqdm
from bs4 import BeautifulSoup

from ratelimit import RateLimitDecorator, limits, sleep_and_retry

ratelimiter = RateLimitDecorator(calls=1, period=1)
requests.get = sleep_and_retry(ratelimiter(requests.get))

# 
# ALL CREDIT FOR WORK IN THIS FILE GOES DIRECTLY TO THE REPOSITORY:
# https://github.com/antiboredom/flickr-scrape
#

KEY = os.environ.get('FLICKR_KEY')
SECRET = os.environ.get('FLICKR_SECRET')

def download_file(url, local_filename):
    if local_filename is None:
        local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return local_filename


def get_group_id_from_url(url):
    params = {
        'method' : 'flickr.urls.lookupGroup',
        'url': url,
        'format': 'json',
        'api_key': KEY,
        'format': 'json',
        'nojsoncallback': 1
    }
    results = requests.get('https://api.flickr.com/services/rest', params=params).json()
    return results['group']['id']


def get_photos(qs, qg, page=1, original=False, bbox=None, num_images=500):
    if num_images < 1: raise Exception('num_images is invalid')
    params = {
        'content_type': '7',
        'per_page': str(min(500, num_images)),
        'media': 'photos',
        'format': 'json',
        'advanced': 1,
        'nojsoncallback': 1,
        'extras': 'media,realname,%s,o_dims,geo,tags,machine_tags,date_taken' % ('url_o' if original else 'url_l'), #url_c,url_l,url_m,url_n,url_q,url_s,url_sq,url_t,url_z',
        'page': page,
        'api_key': KEY
    }

    if qs is not None:
        params['method'] = 'flickr.photos.search',
        params['text'] = qs
    elif qg is not None:
        params['method'] = 'flickr.groups.pools.getPhotos',
        params['group_id'] = qg

    # bbox should be: minimum_longitude, minimum_latitude, maximum_longitude, maximum_latitude
    if bbox is not None and len(bbox) == 4:
        params['bbox'] = ','.join(bbox)

    results = requests.get('https://api.flickr.com/services/rest', params=params).json()
    if "photos" not in results:
        print(results)
        return None
    return results["photos"]


def search(qs, qg, bbox=None, original=False, max_pages=None, start_page=1, output_dir='images', num_images=500, use_subfolder=True):
    # create a folder for the query if it does not exist
    foldername = os.path.join(output_dir, re.sub(r'[\W]', '_', qs if qs is not None else "group_%s"%qg))
    if bbox is not None:
        foldername += '_'.join(bbox)

    if not use_subfolder: foldername = output_dir

    if not os.path.exists(foldername):
        os.makedirs(foldername)

    jsonfilename = os.path.join(foldername, 'results' + str(start_page) + '.json')

    if not os.path.exists(jsonfilename):

        # save results as a json file
        photos = []
        current_page = start_page

        results = get_photos(qs, qg, page=current_page, original=original, bbox=bbox, num_images=num_images)
        if results is None:
            return

        total_pages = results['pages']
        if max_pages is not None and total_pages > start_page + max_pages:
            # total_pages = start_page + max_pages
            total_pages = max_pages

        photos += results['photo']
        downloaded_images = 0
        while downloaded_images < num_images and current_page < total_pages:
            print('downloading metadata, page {} of {}. Stored {} images out of required {}'.format(current_page, total_pages, downloaded_images, num_images))
            temp = get_photos(qs, qg, page=current_page, original=original, bbox=bbox, num_images=num_images)['photo']
            current_page += 1
            
            for photo in temp:
                if downloaded_images < num_images and download_image(photo, foldername, original): downloaded_images += 1
        print('Downloaded {} images after searching through {} pages'.format(downloaded_images, current_page - 1))
        with open(jsonfilename, 'w') as outfile:
            json.dump(photos, outfile)

    else:
        with open(jsonfilename, 'r') as infile:
            photos = json.load(infile)

def download_image(photo, foldername, original):
    # download images
    try:
        url = photo.get('url_o' if original else 'url_l')
        extension = url.split('.')[-1]
        localname = os.path.join(foldername, '{}.{}'.format(photo['id'], extension))

        if not os.path.exists(localname):
            download_file(url, localname)
        return True
    except Exception as e:
        return False


def search_store_query_flickr(query: str, max_pages:int = 1000, num_images:int = 500, dir_name:str = None, dim:tuple = None, use_subfolder:bool = True):
    dir_path = dir_name

    if not dir_path: 
        cur_time = datetime.datetime.now()
        name = 'dataset-' + query + '_' + cur_time.strftime("%m-%d-%YT%H:%M:%S")
        dir_path = str(pathlib.Path(__file__).parent.parent.absolute()) + '/' + name
        
    qs = query
    qg = None
    original = None
    output_dir = dir_path

    if qs is None and qg is None:
        sys.exit('Must specify a search term or group id')

    bbox = dim
    if bbox and len(bbox) != 4:
        bbox = None

    if qg is not None:
        qg = get_group_id_from_url(qg)

    print('Searching for {}'.format(qs if qs is not None else "group %s"%qg))
    if bbox:
        print('Within', bbox)

    start_page = 1

    search(qs, qg, bbox, original, max_pages, start_page, output_dir, num_images=num_images, use_subfolder=use_subfolder)

# search_store_query_flickr('beagle dog', num_images=10)