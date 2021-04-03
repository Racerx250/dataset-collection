from __future__ import print_function
import time
import sys
import json
import re
import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

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


def get_photos(qs, qg, page=1, original=False, bbox=None):
    params = {
        'content_type': '7',
        'per_page': '500',
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


def search(qs, qg, bbox=None, original=False, max_pages=None, start_page=1, output_dir='images'):
    # create a folder for the query if it does not exist
    foldername = os.path.join(output_dir, re.sub(r'[\W]', '_', qs if qs is not None else "group_%s"%qg))
    if bbox is not None:
        foldername += '_'.join(bbox)

    if not os.path.exists(foldername):
        os.makedirs(foldername)

    jsonfilename = os.path.join(foldername, 'results' + str(start_page) + '.json')

    if not os.path.exists(jsonfilename):

        # save results as a json file
        photos = []
        current_page = start_page

        results = get_photos(qs, qg, page=current_page, original=original, bbox=bbox)
        if results is None:
            return

        total_pages = results['pages']
        if max_pages is not None and total_pages > start_page + max_pages:
            total_pages = start_page + max_pages

        photos += results['photo']

        while current_page < total_pages:
            print('downloading metadata, page {} of {}'.format(current_page, total_pages))
            current_page += 1
            photos += get_photos(qs, qg, page=current_page, original=original, bbox=bbox)['photo']
            time.sleep(0.5)

        with open(jsonfilename, 'w') as outfile:
            json.dump(photos, outfile)

    else:
        with open(jsonfilename, 'r') as infile:
            photos = json.load(infile)

    # download images
    print('Downloading images')
    for photo in tqdm(photos):
        try:
            url = photo.get('url_o' if original else 'url_l')
            extension = url.split('.')[-1]
            localname = os.path.join(foldername, '{}.{}'.format(photo['id'], extension))
            if not os.path.exists(localname):
                download_file(url, localname)
        except Exception as e:
            continue

def search_store_query_flickr(search: str, max_pages: int, dir_name: str = None, dim: tuple = None):
    dir_path = dir_name
    if not dir_path: 
        cur_time = datetime.datetime.now()
        name = 'dataset-' + search + '_' + cur_time.strftime("%m-%d-%YT%H:%M:%S")
        dir_path = str(pathlib.Path(__file__).parent.parent.absolute()) + '/' + name
        
    qs = search
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

    search(qs, qg, bbox, original, max_pages, start_page, output_dir)