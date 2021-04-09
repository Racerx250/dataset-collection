import datetime
from image_collector import search_store_query_flickr

cur_time = datetime.datetime.now()
dataset_name = 'dataset_' + 'dogs' + '_' + cur_time.strftime("%m-%d-%YT%H:%M:%S")

classes = [
    'beagle',
    'border collie',
    'miniature schnauzer',
    'boston terrier',
    'labradoodle',
]
for label in classes:
    flickr_dir = dataset_name + '/' + label + '/flickr'
    search_store_query_flickr(label + ' dog', 1, dir_name=flickr_dir, num_images=10, use_subfolder=False)