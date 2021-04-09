from google_images_search import GoogleImagesSearch
from PIL import Image
from io import BytesIO

import numpy as np
import pathlib
import os
import typing
import datetime
import json

gis = GoogleImagesSearch(os.environ.get('GCS_DEVELOPER_KEY'), os.environ.get('GCS_CX'))
my_bytes_io = BytesIO()

def search_store_query_google(search: str, num: int, dir_name: str = None, options:dict = {}) -> None:
    dir_path = dir_name
    if not dir_path: 
        cur_time = datetime.datetime.now()
        name = 'dataset-' + search + '_' + cur_time.strftime("%m-%d-%YT%H:%M:%S")
        dir_path = str(pathlib.Path(__file__).parent.parent.absolute()) + '/' + name

    _search_params = {
        'q': search,
        'num': num,
        'fileType': options.get('fileType', 'jpg'),
        'imgType': options.get('imgType', 'photo'),
        'imgSize': options.get('imgSize', 'LARGE'),
    }

    gis.search(search_params=_search_params, path_to_dir=dir_path, width=options.get('dim', 500), height=options.get('dim', 500))

    try:
        file = open(dir_path + "/query.json", "w")
        json_formatted_str = json.dumps(_search_params, indent=2)
        file.write(json_formatted_str)
        file.close()
    except:
        raise Exception('error writing to file')

    X = []
    for _ in range(int(num / 10) + 1):
        for image in gis.results():
            my_bytes_io.seek(0)
            image.copy_to(my_bytes_io)
            my_bytes_io.seek(0)

            data = np.asarray(Image.open(my_bytes_io))
            X.append(data)
        gis.next_page()
    
    # return np.stack(X, axis=3)
    return X

# search_store_query_google('dog', 2)