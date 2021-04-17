import datetime
from image_manager import search_store_query_flickr
from image_manager import search_store_query_google
import time

cur_time = datetime.datetime.now()
dataset_name = 'dataset_' + 'dogs' + '_' + 'small_dirty'

classes = [
    'border collie dog',
    'miniature schnauzer dog',
]

label = classes[0]
print('Starting Flickr scrape of ' + label + '.')
flickr_dir = dataset_name + '/' + label.replace(' ', '_') + '/flickr'
search_store_query_flickr(label, dir_name=flickr_dir, num_images=100, use_subfolder=False)

label = classes[1]
print('Starting Flickr scrape of ' + label + '.')
flickr_dir = dataset_name + '/' + label.replace(' ', '_') + '/flickr'
search_store_query_flickr(label, dir_name=flickr_dir, num_images=10, use_subfolder=False)

