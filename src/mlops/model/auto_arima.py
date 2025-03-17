from src.mlops.abstractions import UnivariateTSModelABC
from typing import Union, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import pmdarima as pm
from autots import AutoTS
import warnings
warnings.filterwarnings("ignore")

class ARIMA(UnivariateTSModelABC):
    """
    Forecast future values using an automatic ARIMA model.

    Parameters:
    - training_points (array-like): Historical time series data used to train the ARIMA model.
    - forecast_points (int): The number of future data points to forecast.

    Returns:
    - list: A list containing the forecasted values.
    """
    def train(self, training_points: pd.Series, forecast_points: int) -> list:
        # Fit the Auto ARIMA model
        try:
            # Fit the Auto ARIMA model
            model = pm.auto_arima(
                y=training_points,
                X=None,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True
            )

            # Forecast future points
            forecast, confint = model.predict(n_periods=forecast_points, return_conf_int=True)

            # Convert forecast results to list and return
            return forecast.tolist()

        except Exception as e:
            print(f"ARIMA model failed with error: {e}")
            print("Using the last known value for all forecast points.")

            # Use the last known value for all forecast points
            if len(training_points) > 0:
                last_value = training_points[-1]
            else:
                last_value = 0  # Default to zero if training_pipeline data is empty

            forecast = [last_value] * forecast_points
            return forecast



