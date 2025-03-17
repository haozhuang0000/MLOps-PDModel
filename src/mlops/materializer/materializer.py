# import os
# import pickle
# from typing import Any, Type, Union, Tuple, Dict, List
#
# import numpy as np
# import pandas as pd
# from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor
# from sklearn.ensemble import RandomForestRegressor
# # from xgboost import XGBRegressor
# from zenml.io import fileio
# from zenml.materializers.base_materializer import BaseMaterializer
#
# DEFAULT_FILENAME = "CustomerSatisfactionEnvironment"
# class CSMaterializer(BaseMaterializer):
#     """
#     Custom materializer for the Customer Satisfaction Project
#     """
#
#     ASSOCIATED_TYPES = (
#         str,
#         dict,
#         list,
#         tuple,
#         np.ndarray,
#         pd.Series,
#         pd.DataFrame,
#         # CatBoostRegressor,
#         RandomForestRegressor,
#         LGBMRegressor,
#         # XGBRegressor,
#     )
#
#     def handle_input(
#         self, data_type: Type[Any]
#     ) -> Union[
#         str,
#         dict,
#         list,
#         tuple,
#         np.ndarray,
#         pd.Series,
#         pd.DataFrame,
#         RandomForestRegressor,
#         LGBMRegressor,
#     ]:
#         """
#         Loads the model or data from the artifact and returns it.
#
#         Args:
#             data_type: The type of the model or data to be loaded
#         """
#         super().handle_input(data_type)
#         filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
#         with fileio.open(filepath, "rb") as fid:
#             obj = pickle.load(fid)
#         return obj
#
#     def handle_return(
#         self,
#         obj: Union[
#             str,
#             dict,
#             list,
#             tuple,
#             np.ndarray,
#             pd.Series,
#             pd.DataFrame,
#             RandomForestRegressor,
#             LGBMRegressor,
#         ],
#     ) -> None:
#         """
#         Saves the model or data to the artifact store.
#
#         Args:
#             obj: The object (model or data) to be saved
#         """
#         super().handle_return(obj)
#         filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
#         with fileio.open(filepath, "wb") as fid:
#             pickle.dump(obj, fid)
