import os
import random
from tqdm import tqdm
import warnings
import torch
from models.utils import load_pkl, load_json, save_pkl
from models.feature_extractor import FeatureExtractor
from config import BaseConfig

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    CONFIG = BaseConfig().get_args()
    random.seed(CONFIG.seed_num)
    device = torch.device("cuda")
    feature_extractor = FeatureExtractor(device, CONFIG.biobert_path)

    TRAIN_PATH = CONFIG.train_path
    TEST_PATH = CONFIG.test_path
    DEV_PATH = CONFIG.dev_path

    TRAIN_DATA = load_json(TRAIN_PATH)
    TEST_DATA = load_json(TEST_PATH)
    DEV_DATA = load_json(DEV_PATH)

    TRAIN_REL = [[example['tail']['word'], example['head']['word'], example['relation']]
                 for example in tqdm(TRAIN_DATA)][:1000]

    DEV_REL = [[example['tail']['word'], example['head']['word'], example['relation']]
               for example in tqdm(DEV_DATA)][:1000]

    TEST_REL = [[example['tail']['word'], example['head']['word'], example['relation']]
                for example in tqdm(TEST_DATA)][:1000]

    random.shuffle(TRAIN_REL)
    random.shuffle(DEV_REL)
    random.shuffle(TEST_REL)

    x_train, y_train = feature_extractor.transform(TRAIN_REL, train=False)
    x_dev, y_dev = feature_extractor.transform(DEV_REL, train=False)
    x_test, y_test = feature_extractor.transform(TEST_REL, train=False)

    print(f"NUMBER OF RELATIONS:{len(feature_extractor.get_relations())}")

    print(f"SAVE LABEL ENCODER MODEL INTO :{CONFIG.pre_trained_dir} DIR")
    feature_extractor.save_label_encoder(CONFIG.pre_trained_dir)
    
    dataset = {
        "x-train": x_train, "y-train":y_train, 
        "x-test":x_test, "y-test":y_test,
        "x-dev": x_dev, "y-dev":y_dev
    }
    print(f"saving data into the {CONFIG.pre_trained_dir}")
    save_pkl(CONFIG.path_biobert_dataset, dataset)
