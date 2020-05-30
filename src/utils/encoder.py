from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

class FakeRealEncoder:
    def __init__(self):
        self.le = LabelEncoder()
        self.labels = ['FAKE','REAL']
        self.le.fit(self.labels)

    def to_categorical(self, y):
        return np_utils.to_categorical(self.le.transform(y),2)

    def inverse_transform(self, y):
        return self.le.inverse_transform(y[:,1].astype(int))