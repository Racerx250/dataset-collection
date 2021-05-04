import datetime
from image_manager import search_store_query_flickr
from image_manager import search_store_query_google
import time

cur_time = datetime.datetime.now()
dataset_name = 'dataset_' + 'dogs' + '_' + cur_time.strftime("%m-%d-%YT%H:%M:%S")

classes = open('image_manager/stanford_dogs_labels.txt').readlines()
classes = [e.replace('\n', '').lower() for e in temp]
# classes = [
#     'beagle',
#     'border collie',
#     'miniature schnauzer',
#     'boston terrier',
#     'labradoodle',
#     'chihuahua',
#     'golden retriever',
#     'french bulldog',
#     'poodle',
#     'pembroke welsh corgi',
#     'german shepherd',
#     'great dane',
#     'shiba inu',
#     'dalmatian',
#     'labrador retriever'
# ]
for label in classes:
    print('Starting Flickr scrape of ' + label + '.')
    flickr_dir = dataset_name + '/' + label.replace(' ', '_') + '/flickr'
    search_store_query_flickr(label + (' dog' if 'dog' not in label else ''), dir_name=flickr_dir, num_images=150, use_subfolder=False)

    # print('Starting Google scrape of ' + label + '.')
    # google_dir = dataset_name + '/' + label.replace(' ', '_') + '/google'
    # search_store_query_google(label + ' dog', 100, dir_name=google_dir)
