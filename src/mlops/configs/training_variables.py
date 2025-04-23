from src.mlops.configs import LGBMTrainingConfig
from abc import ABC, abstractmethod

class TrainingVariables:

    training_config = LGBMTrainingConfig()