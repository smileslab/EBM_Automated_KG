from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.layers import SpatialDropout1D, Conv1D, Input
from tensorflow.keras.layers import MaxPooling1D, GRU, BatchNormalization
from tensorflow.keras.layers import Input, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D, concatenate, LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class RelationExtractor:
    def __init__(self, f1, checkpoint_path="relation-extraction", batch_size=8,
                 filters=100, lstm_units=50, kernel_size=2, relation_no=2):

        self.model = Sequential([
            SpatialDropout1D(0.5),
            Conv1D(filters, kernel_size=kernel_size, kernel_regularizer=regularizers.l2(0.00001), padding='same'),
            LeakyReLU(alpha=0.2),
            MaxPooling1D(pool_size=2),
            Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True)),

            SpatialDropout1D(0.5),
            Conv1D(filters, kernel_size=kernel_size, kernel_regularizer=regularizers.l2(0.00001), padding='same'),
            LeakyReLU(alpha=0.2),
            MaxPooling1D(pool_size=2),
            Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(lstm_units)),

            Dense(relation_no, activation='softmax')
        ])

        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path + "/cp-{epoch:04d}.ckpt"
        self.cp_callback = ModelCheckpoint(filepath=self.checkpoint_path,
                                           save_weights_only=True,
                                           save_freq=1000 * self.batch_size,
                                           verbose=1)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1])

    def fit(self, X_train, y_train, X_val, y_val, epochs):
        self.history = self.model.fit(X_train, y_train,
                                      batch_size=self.batch_size,
                                      epochs=epochs,
                                      validation_data=[X_val, y_val],
                                      callbacks=[self.cp_callback])

    def get_history(self):
        return self.history

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model

    def evaluate(self, X, y):
        print("EVALUATION RESULTS:", self.model.evaluate(X, y, verbose=1))

    def load_model(self, checkpoint_dir="relation-extraction/", latest=True):
        if latest:
            self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        else:
            self.model.load_weights(checkpoint_dir)
