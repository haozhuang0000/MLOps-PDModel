# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:35:12 2025

@author: nixingyu
@modified by zhuanghao
"""
import time
import smbclient
import scipy.io
import io
import pandas as pd
import os
import math
from dotenv import load_dotenv
load_dotenv()

def get_XY(x_path: str, y_path: str):
    # Setup SMB session (use correct IP/hostname and credentials)
    smbclient.register_session(
        os.getenv("FILEIP"),
        username=os.getenv("FILEIP_USERNAME"),
        password=os.getenv("FILEIP_PASSWORD")
    )

    # base_dir = fr"\\{os.getenv('FILEIP')}\cri3\OfficialTest_AggDTD_SBChinaNA\ProductionData\Recent\Daily"
    # all_files = smbclient.listdir(base_dir)
    # daily_folder = [f for f in all_files if f.startswith(f"{date}_cali_")]
    #
    # if not daily_folder:
    #     raise FileNotFoundError(f"No folder found for {date}")
    #
    # folder_path = os.path.join(base_dir, daily_folder[0])
    #
    # # Build x_path and read .mat content via SMB
    # x_path = os.path.join(folder_path, r"Processing\P2_Pd\covariates\final", f"DATA_{econ}.mat")
    with smbclient.open_file(x_path, mode='rb') as f:
        x_bytes = f.read()
        inputs = scipy.io.loadmat(io.BytesIO(x_bytes))['firmspecific']

    reshaped = inputs.transpose(0, 2, 1).reshape(-1, 21)
    X = pd.DataFrame(reshaped)
    # X.columns = ['Comp_No', 'YYYY', 'MM', 'Stock_Index_Return', 'Three_Month_Rate_After_Demean',
    #              'DTD_Level', 'DTD_Trend', 'Liquidity_Level_Nonfinancial', 'Liquidity_Trend_NonFinancial',
    #              'NI_Over_TA_Level', 'NI_Over_TA_Trend', 'Size_Level', 'Size_Trend', 'M_Over_B',
    #              'SIGMA', 'Liquidity_Level_Financial', 'Liquidity_Trend_Financial', 'DTD_Median_Fin',
    #              'DTD_Median_Nfin', 'dummy_for_China_SOE', 'dummy_for_NM']
    X.columns = ['Comp_No', 'YYYY', 'MM', 'StkIndx', 'STInt',
                 'dtdlevel', 'dtdtrend', 'liqnonfinlevel', 'liqnonfintrend',
                 'ni2talevel', 'ni2tatrend', 'sizelevel', 'sizetrend', 'm2b',
                 'sigma', 'liqfinlevel', 'lqfintrend', 'DTDmedianFin',
                 'DTDmedianNonFin', 'dummySOE', 'dummyNM']

    # # Now load y_path (FirmHistory file)
    # y_path = os.path.join(folder_path, r"IDMTData\Smart\FirmHistory", f"FirmHistory_{econ}.mat")
    with smbclient.open_file(y_path, mode='rb') as f:
        y_bytes = f.read()
        y = pd.DataFrame(scipy.io.loadmat(io.BytesIO(y_bytes))['firmHistory'])

    y = y.iloc[:, [0, 2, 4]]
    y.columns = ['Comp_No', 'YYYYMMDD', 'Event_Type']
    y['Comp_No'] = y['Comp_No'].apply(lambda X: math.floor(X / 1000))
    y['Event_Type'] = y['Event_Type'].fillna(0)

    return X, y