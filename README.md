# dataset-collection
UC San Diego research project

### Image Collector
Queries for the given terms, stores in a given directory, and then returns a list of numpy arrays where each numpy array is a 3D representation of an image (height-width-RGB). 

Main function is given by:

`def search_store_query(search: str, num: int, dir_name: str = None, options:dict = {}) -> None:`

Where:
- *search* is the string you would input into Google to find the images
- *num* is the amount of images
- *dir_name* is the directory you would like to save in (will generate a directory if not given)
- *options* is misc. options like fileType, imgType, imgSize, etc. See code to become familiar

#### Google Image Searches
Max queries per day is 100 without a pricing plan. Above this, it's $5 per 1000 queries. We may be able to get this funding from the department.
Ping lclawren@eng.ucsd.edu for API and CX key, or just message me in Slack. Then, you can add the keys to your local enviroment using the following commands (for linux):

`export GCS_DEVELOPER_KEY = *insert developer key`
`export GCS_CX = *insert index key`

### NOTES
Easy installation:

`pip install -r requirements.txt`

