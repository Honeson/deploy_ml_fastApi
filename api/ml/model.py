import joblib
import numpy as np
import pandas as pd
from pathlib import Path

#from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
#from sklearn.datasets import load_boston


class Model:
    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        self.load()


    def train(self, X: np.ndarray, y: np.ndarray):
        self._model = Lasso(alpha=0.001, random_state=0)
        self._model.fit(X,y)
        return self


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict(X_test)


    def save(self):
        if self._model is not None:
            joblib.dump(self._model, self._model_path)
        else:
            raise TypeError('The model is not trained yet. Please train the model before saving')


    def load(self):
        try:
            self._model = joblib.load(self._model_path)
        except:
            self._model = None
        return self
root_path = Path(__file__).parent
model_path = root_path/'model.joblib'
file_path = root_path/'data'
X_train = pd.read_csv(file_path/'xtrain.csv')
X_test = pd.read_csv(file_path/'xtest.csv')
y_train = pd.read_csv(file_path/'ytrain.csv')
y_test = pd.read_csv(file_path/'ytest.csv')
n_features = X_train.shape[1]
model = Model(model_path)

def get_model():
    return model

features = pd.read_csv(file_path/'selected_features.csv')
features = features['0'].to_list()
X_train = X_train[features]
X_test= X_test[features]


if __name__ == '__main__':
    model.train(X_train, y_train)
    model.save()

    print(X_train.head())
    

    print(y_test[:20])
    pred = model.predict(X_test)
    print(pred[:20])