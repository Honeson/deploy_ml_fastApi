from typing import List
from fastapi import Depends, FastAPI
import numpy as np
from pydantic import BaseModel, ValidationError, validator

from .ml.model import get_model, Model, n_features


app = FastAPI()


class Predictrequest(BaseModel):
    data: List[List[float]]

    @validator('data')
    def check_dimentionality(cls, list_of_lists):
        for list_of_data in list_of_lists:
            if len(list_of_data) != n_features:
                raise ValueError(f'Each list of data points in the list must contain {n_features} features. You provided {len(list_of_data)}')
        return list_of_lists


class Predictresponse(BaseModel):
    data: List[float]


@app.post('/predict/', response_model=Predictresponse)
def predict(input: Predictrequest, model: Model = Depends(get_model)):
    X_test = np.array(input.data)
    y_pred = model.predict(X_test)
    response = Predictresponse(data=y_pred.tolist())
    return response