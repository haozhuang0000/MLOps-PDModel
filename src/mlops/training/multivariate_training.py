from src.mlops.abstractions import TrainingABC
from src.mlops.model import ModelCaller, RFRegression, LGBRegression, H2OAuto
from typing import Union, Tuple, List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

class MultivariateTrainingHelper:

    def __init__(self,
                 df_X_train: pd.DataFrame,
                 df_y_train: pd.Series,
                 id_bb_unique: str,
                 y: str,
                 model: Union[RFRegression, LGBRegression]):

        self.df_X_train = df_X_train
        self.df_y_train = df_y_train
        self.id_bb_unique = id_bb_unique
        self.y = y
        self.model = model

    def run(self):

        model, model_name, model_run_id = ModelCaller(strategy=self.model).train(
                                                    X_train=self.df_X_train,
                                                    y_train=self.df_y_train,
                                                    id_bb_unique=self.id_bb_unique,
                                                    y=self.y)
        return model, model_name, model_run_id

class MultivariateTraining(TrainingABC):

    def run_training(self,
                     single_company_df_dict: Dict,
                     id_bb_unique: str,
                     y: str,
                     model: Union[RFRegression, LGBRegression]):

        if isinstance(model, H2OAuto):
            key_name = 'df_train_w_category'
        else:
            key_name = 'df_train'

        df_train = single_company_df_dict[key_name]
        df_y_train = df_train[y]
        df_X_train = df_train.drop(columns=[y])
        model, model_name, model_run_id = MultivariateTrainingHelper(df_X_train=df_X_train,
                                                                     df_y_train=df_y_train,
                                                                     id_bb_unique=id_bb_unique,
                                                                     y=y,
                                                                     model=model).run()
        return model, model_name, model_run_id

# class MultivariateTraining(TrainingABC):
#
#     def run_training(self, companyID_df_postConstru_dict: Dict, model: Union[RFRegression]) -> Dict:
#
#         y_cols = self.y_cols
#
#         for y in y_cols[:1]:
#             for id_bb_unique, single_company_df_dict in companyID_df_postConstru_dict.items():
#                 res = MultivariateTrainingHelper().run_by_company(single_company_df_dict, id_bb_unique, y, model)
#                 break
