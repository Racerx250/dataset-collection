import datetime
from query_flickr import search_store_query_flickr

cur_time = datetime.datetime.now()
name = 'dataset_' + 'dogs' + '_' + cur_time.strftime("%m-%d-%YT%H:%M:%S") + '/border_collie/flickr'
search_store_query_flickr('collie', 1, dir_name=name, num_images=10)