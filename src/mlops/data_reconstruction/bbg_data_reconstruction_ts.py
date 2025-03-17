from src.mlops.abstractions import DataReconstructionStrategyABC
from src.mlops.model import ModelCaller, AR, ARIMA
from typing import Union, Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm

class BBGDataReconstructionTSHelper:

    @staticmethod
    def data_reconstruction_helper(df, target_column, model):
        """
        Processes a DataFrame containing time series data, segments it into individual series,
        and applies a forecasting model to each series.

        Parameters:
        - df (DataFrame): The input DataFrame containing the time series data.
        - target_column (str): The name of the target column in the DataFrame to forecast.
        - model (function): A forecasting function that takes in training_pipeline data and the number of points to forecast.

        Returns:
        - result (list): A list of forecasted values for each series.
        - result_flatten (list): A flattened list of all forecasted values.
        - training_data_with_forecast_result (list): A combined list of training_pipeline data and forecasted values for all series.
        """
        training_data = []
        forecasting_point = 0
        result = []
        training_data_with_forecast_result = []

        for index, row in df.iterrows():
            if pd.notna(row[target_column]):
                training_data.append(row[target_column])
            else:
                forecasting_point += 1

        if len(training_data) > 3:
            # For the final timeseries
            forecast_values = ModelCaller(strategy=model).train(training_points=training_data, forecast_points=forecasting_point)
            result.append(forecast_values)
            training_data_with_forecast_result.append(training_data)
            training_data_with_forecast_result.append(forecast_values)

            result_flatten = [item for sublist in result for item in sublist]
            training_data_with_forecast_result = [item for sublist in training_data_with_forecast_result for item in sublist]
        else:
            forecast_values = [training_data[-1] for _ in range(forecasting_point)]
            result.append(forecast_values)
            training_data_with_forecast_result.append(training_data)
            training_data_with_forecast_result.append(forecast_values)

        result_flatten = [item for sublist in result for item in sublist]
        return result_flatten, training_data, training_data_with_forecast_result



class BBGDataReconstructionTS(DataReconstructionStrategyABC):

    def handle_data(self, id_bb_unique: str,
                    df_dict: dict,
                    model: Union[AR, ARIMA],
                    modify_dict_type: str) -> dict:

        # companyID_df_postConstru_dict = {}
        # for id_bb_unique, df_dict in tqdm(companyID_df_dict.items(),
        #                        desc=f"running data reconstruction - calling: {BBGDataReconstructionTS().__class__.__name__} - model: **{model.__class__.__name__}**"):
        """
        df_dict with key - value: {
            'df_preConstru': sub_df,
            'df_train': sub_df_train,
            'df_test': df_test,
            'nan_rows': nan_rows,
        }
        """
        training_data_dict = {}
        result_flatten_dict = {}
        df_x = df_dict["df_preconstru"][self.x_cols_to_process]
        for col in self.x_cols_to_process:
            if col in df_x.columns:
                result_flatten, training_data, training_data_with_forecast_result = \
                    BBGDataReconstructionTSHelper.data_reconstruction_helper(df_x, col, model)
                training_data_dict[col] = training_data
                result_flatten_dict[col] = result_flatten

        ## data after reconstruction
        # df_forecast_x = pd.DataFrame(forecast_results)

        ## training data
        df_train = pd.DataFrame(training_data_dict)

        df_groud_truth_data = df_dict['df_ground_truth'][self.y_cols].reset_index(drop=True)
        nan_rows = df_groud_truth_data[df_groud_truth_data.isnull().any(axis=1)].index.tolist()

        ## test data predicted by univariate ts model - float
        df_test = pd.DataFrame(result_flatten_dict)

        ## test data - float + category
        df_test_w_category = df_test.copy()
        df_test_w_category[self.industry_info_cols] = df_dict['df_test'][self.industry_info_cols].reset_index(drop=True)

        df_test_deploy = df_test.copy()
        df_test_w_category_deploy = df_test_w_category.copy()

        df_test = df_test.drop(nan_rows)
        df_test_w_category = df_test_w_category.drop(nan_rows)

        if modify_dict_type == 'create':
            temp_dict = {
                'df_preconstru': df_dict['df_preconstru'],
                'df_ground_truth': df_dict['df_ground_truth'],
                'df_test': df_dict['df_test'],
                'df_train': df_train,
                'df_train_w_category': df_dict['df_train'],
                'df_y_test': df_groud_truth_data.drop(nan_rows),
                # 'nan_rows': df_dict['nan_rows'],
                # f'df_x_{model.__class__.__name__}_postConstru'.lower(): df_forecast_x,
                f'df_test_predby_{model.__class__.__name__}'.lower(): df_test,
                f'df_test_predby_{model.__class__.__name__}_w_category'.lower(): df_test_w_category,
                f'df_test_predby_{model.__class__.__name__}_deploy'.lower(): df_test_deploy,
                f'df_test_predby_{model.__class__.__name__}_deploy_w_category'.lower(): df_test_w_category_deploy
            }
        elif modify_dict_type == 'append':
            temp_dict = df_dict
            # temp_dict[f'df_x_{model.__class__.__name__}_postConstru'.lower()] = df_forecast_x
            temp_dict[f'df_test_predby_{model.__class__.__name__}'.lower()] = df_test
            temp_dict[f'df_test_predby_{model.__class__.__name__}_w_category'.lower()] = df_test_w_category
            temp_dict[f'df_test_predby_{model.__class__.__name__}_deploy'.lower()] = df_test_deploy
            temp_dict[f'df_test_predby_{model.__class__.__name__}_deploy_w_category'.lower()] = df_test_w_category_deploy
        else:
            raise Exception(f"modify_dict_type must be 'create' or 'append'")
        # companyID_df_postConstru_dict[id_bb_unique] = temp_dict

        return id_bb_unique, temp_dict