from src.mlops.abstractions import UnivariateTSModelABC
from typing import Union, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.api as sm

class AR(UnivariateTSModelABC):
    """
    Forecast future values using an AR(1) (autoregressive) model.

    Parameters:
    - training_points (list or array-like): Historical time series data used to train the AR(1) model.
    - forecast_points (int): The number of future data points to forecast.

    Returns:
    - list: A list containing the forecasted values.
    """
    lag = 1

    def train(self, training_points: pd.Series, forecast_points: int) -> list:

        """
        Forecast future values using an AR(1) (autoregressive) model.

        Parameters:
        - training_points (list or array-like): Historical time series data used to train the AR(1) model.
        - forecast_points (int): The number of future data points to forecast.

        Returns:
        - list: A list containing the forecasted values.
        """
        # Convert the list of training_pipeline points to a numpy array
        training_points = np.array(training_points)

        # Fit the AR(1) model using statsmodels
        ar_model = sm.tsa.AutoReg(training_points, lags=self.lag).fit()

        # Get the initial training_pipeline series for forecasting
        start_point = len(training_points)
        end_point = start_point + forecast_points - 1

        # Use the fitted AR model to make forecasts
        forecast = ar_model.predict(start=start_point, end=end_point)

        # Convert forecast results to list and return
        return forecast.tolist()