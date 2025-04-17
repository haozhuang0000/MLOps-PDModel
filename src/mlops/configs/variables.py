from src.mlops.configs import XYVariables, BBGFields
from abc import ABC, abstractmethod

class Variables:

    xyvariables = XYVariables()
    bbgfields = BBGFields()

    x_cols_to_process_cn = xyvariables.X_SELECTED_CN
    y_cols = xyvariables.Y_SELECTED
    industry_cols = xyvariables.INDUSTRY
    industry_info_cols = xyvariables.INDUSTRY_TIC_INFO

    # Define the priority mapping for 'FILING_STATUS'
    filing_status_priority = bbgfields.FILING_STATUS_PRIORITY
    # Define the priority mapping for 'ACCOUNTING_STANDARD'
    accounting_standard_priority = bbgfields.ACCOUNTING_STANDARD_PRIORITY

    expand_year = 2030