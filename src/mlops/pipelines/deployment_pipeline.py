import json
import os
from pickle import FALSE
import numpy as np
import pandas as pd
from src.mlops.steps_deployment import (
    load_data,
    load_registered_model,
    predict
)
import pandas as pd

def prediction_service(
        model_name:str,
        id_bb_unique: str,
        y: str,
        year: int
):
    X = load_data(id_bb_unique, y, year)
    model = load_registered_model(model_name, id_bb_unique, y)
    prediction = predict(model, X)
    print(X)
    print(prediction)
    return prediction

if __name__ == "__main__":
    prediction_service('LGBRegression', 'EQ0000000000142041', 'EBITDA', 2027)