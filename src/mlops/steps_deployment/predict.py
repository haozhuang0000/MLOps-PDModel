from typing import Tuple, Union, Dict, Any
from typing_extensions import Annotated
import pandas as pd
import numpy as np
import pickle
import os
from zenml import step
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker

# @step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def predict(model: Any, X: pd.DataFrame) -> float:
    prediction = model.predict(X)
    return prediction[0]