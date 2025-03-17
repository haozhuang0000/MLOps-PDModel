from src.mlops.configs import TrainingConfig
from abc import ABC, abstractmethod

class TrainingVariables:

    training_config = TrainingConfig()
    x_reconstruction_type = training_config.X_RECONSTRUCTION_TYPE