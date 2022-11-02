from transformers import BertTokenizer, BertModel
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import os
from tqdm import tqdm

class FeatureExtractor:
    """
        https://github.com/dmis-lab/biobert-pytorch
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """

    def __init__(self, device, path):
        self.device = device
        self.biobert_tokenizer = BertTokenizer.from_pretrained(path)
        self.biobert_model = BertModel.from_pretrained(path)
        self.biobert_model.to(self.device)
        self.label_encoder = LabelEncoder()

    def get_label_encoder(self):
        return self.label_encoder

    def load_label_encoder(self, path):
        with open(path + "/label-encoder.sav", "rb") as myfile:
            self.label_encoder = pickle.load(myfile)

    def save_label_encoder(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(path + "/label-encoder.sav", "wb") as myfile:
            pickle.dump(self.label_encoder, myfile, protocol=pickle.HIGHEST_PROTOCOL)

    def transform(self, X, train=True):
        Xs, y = [], []
        for x in tqdm(X):
            Xs.append(self.transform_pairs(x[0], x[1]))
            y.append(x[2])
        if train:
            self.label_encoder.fit(y)
        X_data = np.array(Xs).reshape(len(Xs), Xs[0].shape[0], 1)
        y_data = to_categorical(self.label_encoder.transform(np.array(y)), num_classes=len(self.get_relations()))
        return X_data, y_data

    def transform_pairs(self, ent1, ent2):
        ent1_vector = self.get_features(ent1)
        ent2_vector = self.get_features(ent2)
        return np.concatenate((ent1_vector, ent2_vector), axis=0)

    def get_features(self, entity):
        inputs = self.biobert_tokenizer(entity, return_tensors="pt",
                                        padding=True, truncation=True)
        inputs.to(self.device)
        outputs = self.biobert_model(**inputs)[0][0].cpu().detach().numpy()
        avg = sum(outputs) / outputs.shape[0]
        return avg

    def get_relations(self):
        return self.label_encoder.classes_