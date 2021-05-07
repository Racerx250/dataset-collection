import pathlib
import sys

PROJECT_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.insert(0, PROJECT_DIR)

from clean_dataset import clean_dataset_dir

from query_google import search_store_query_google
from query_flickr import search_store_query_flickr

from database_interface import get_database, create_dirty_database, DatabaseInterface, DirtyDatabase