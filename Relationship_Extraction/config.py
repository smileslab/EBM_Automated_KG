"""
    BaseConfig
         Get model parameters
"""
import argparse
from pathlib import Path
import os


class BaseConfig:
    """
        Base Configigurations
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # dataset and dataset directory
        self.parser.add_argument("--train_path", type=str,
                                 default="data/train.json",
                                 help='Path to train file')
        self.parser.add_argument("--test_path", type=str,
                                 default="data/test.json",
                                 help='Path to test file')
        self.parser.add_argument("--dev_path", type=str,
                                 default="data/dev.json",
                                 help='Path to dev file')
        self.parser.add_argument("--biobert_path", type=str,
                                 default="assets/transformers/biobert-base-cased-v1.1",
                                 help='path to bio bert')
        self.parser.add_argument("--distilbert_path", type=str,
                                 default="assets/transformers/distilbert-base-uncased",
                                 help='path to distilbert')
        self.parser.add_argument("--pre_trained_dir", type=str,
                                 default="assets/model/",
                                 help='path to the pretrained model directory?')
        self.parser.add_argument("--path_biobert_dataset", type=str,
                                 default="assets/model/biobert_dataset.pkl",
                                 help='path to the transformed and saved dataset')
        self.parser.add_argument("--path_saved_le", type=str,
                                 default="assets/model/label-encoder.sav",
                                 help='path to the transformed and saved dataset')
        self.parser.add_argument("--seed_num", type=int, default=222,
                                 help="Default Seed Num to regenerate results")
        self.parser.add_argument("--batch_size", type=int, default=8, help='Batch Size')
        self.parser.add_argument("--epochs", type=int, default=3, help='Epoch Number')
        self.parser.add_argument("-f")
        
    def get_args(self):
        """
            Return parser
        :return: parser
        """
        return self.parser.parse_args()