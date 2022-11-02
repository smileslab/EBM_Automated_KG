import json
import pickle

def save_json(path, data):
    '''save json data into specified path'''
    json.dump(data, open(path, "w"), indent=4)

def load_json(path):
    '''load json data from specified path'''
    return json.load(open(path, "r"))

def load_pkl(path):
    with open(path, "rb") as f:
        pkl = pickle.load(f)
    return pkl

def save_pkl(path, data):
    '''save pickle file'''
    with open(path, "wb") as myfile:
        pickle.dump(data, myfile, protocol=pickle.HIGHEST_PROTOCOL)
