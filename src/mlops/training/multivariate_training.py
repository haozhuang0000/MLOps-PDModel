from src.mlops.abstractions import TrainingABC
from src.mlops.model import ModelCaller, RFRegression, LGBRegression, LGBClassifier
from typing import Union, Tuple, List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

class MultivariateTrainingHelper:

    def __init__(self,
                 train_val_splits: List[Dict[str, pd.DataFrame]],
                 train_df: pd.DataFrame,
                 model: Union[RFRegression, LGBRegression]):

        self.train_val_splits = train_val_splits
        self.train_df = train_df
        self.model = model

    def run(self):

        model, model_name, model_run_id = ModelCaller(strategy=self.model).train(train_val_splits=self.train_val_splits,
                                                                                 train_df=self.train_df)
        return model, model_name, model_run_id

class MultivariateTraining(TrainingABC):

    def run_training(self,
                     train_val_splits: List[Dict[str, pd.DataFrame]],
                     train_df: pd.DataFrame,
                     model: Union[LGBClassifier]):

        model, model_name, model_run_id = MultivariateTrainingHelper(train_val_splits=train_val_splits,
                                                                     train_df=train_df,
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
