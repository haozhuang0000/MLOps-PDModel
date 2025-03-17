import os

class TrainingConfig:
    def __init__(self):
        self._X_RECONSTRUCTION_TYPE = [
            'ar',
            'arima'
        ]

    @property
    def X_RECONSTRUCTION_TYPE(self):
        return self._X_RECONSTRUCTION_TYPE