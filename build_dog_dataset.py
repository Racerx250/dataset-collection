import datetime
from image_collector import search_store_query_flickr
from image_collector import search_store_query_google

cur_time = datetime.datetime.now()
dataset_name = 'dataset_' + 'dogs' + '_' + cur_time.strftime("%m-%d-%YT%H:%M:%S")

classes = [
    'beagle',
    'border collie',
    'miniature schnauzer',
    'boston terrier',
    'labradoodle',
    'chihuahua',
    'golden retriever',
    'french bulldog',
    'poodle',
    'pembroke welsh corgi',
    'german shepherd',
    'great dane',
    'shiba inu',
    'dalmatian',
    'labrador retriever'
]
for label in classes:
    print('Starting Flickr scrape of ' + label + '.')
    flickr_dir = dataset_name + '/' + label.replace(' ', '_') + '/flickr'
    search_store_query_flickr(label + ' dog', dir_name=flickr_dir, num_images=80, use_subfolder=False)
    
    print('Starting Google scrape of ' + label + '.')
    google_dir = dataset_name + '/' + label.replace(' ', '_') + '/google'
    search_store_query_google(label + ' dog', 80, dir_name=google_dir)
