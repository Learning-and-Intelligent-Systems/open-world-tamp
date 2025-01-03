import random

TEMP_DIR = "temp_meshes/"


def safe_sample(collection, k=1):
    collection = list(collection)
    if len(collection) <= k:
        return collection
    return random.sample(collection, k)
