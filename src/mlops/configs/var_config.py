import os

class BBGFields:
    def __init__(self):
        self._FILING_STATUS_PRIORITY = {
            'MR': 1,
            'OR': 2,
            'PR': 3,
            'RS': 4,
            'ER': 5
        }
        self._ACCOUNTING_STANDARD_PRIORITY = {
            'IAS/IFRS': 1,
            'US GAAP': 2
        }

    @property
    def FILING_STATUS_PRIORITY(self):
        return self._FILING_STATUS_PRIORITY

    @property
    def ACCOUNTING_STANDARD_PRIORITY(self):
        return self._ACCOUNTING_STANDARD_PRIORITY

class XYVariables:
    def __init__(self):
        # self._X_SELECTED = [
        #     'BS_TOT_LIAB2', 'BS_TOT_ASSET', 'BS_ACCT_PAYABLE', 'BS_PREPAY',
        #     'BS_ACCTS_REC_EXCL_NOTES_REC', 'BS_INVENTORIES', 'TOTAL_EQUITY',
        #     'CF_DEPR_AMORT', 'CF_DVD_PAID', 'CF_DEPR_EXP', 'CF_CASH_FROM_OPER',
        #     'CF_DEF_INC_TAX', 'WORKING_CAPITAL', 'TAX_BURDEN', 'RETURN_ON_ASSET',
        #     'PROF_MARGIN', 'CHNG_WORK_CAP', 'SALES_REV_TURN', 'NET_OPER_INCOME',
        #     'IS_OPER_INC', 'IS_OPERATING_EXPN', 'IS_INC_TAX_EXP', 'IS_INT_EXPENSE',
        #     'EFF_INT_RATE', 'IS_PROVISION_DOUBTFUL_ACCOUNTS', 'EBITDA',
        #     'ARD_FOREIGN_EXCHANGE_GAIN_LOSS', 'ARD_STOCK_BASED_COMPENSATION',
        #     'ARD_EFF_OF_EXCH_RATES_ON_CASH', 'ARD_OTHER_NON_CASH_ITEMS',
        #     'ARD_GAIN_LOSS_SALES_FIXED_ASSETS', 'ARD_CAPITAL_EXPENDITURES',
        #     'ARD_COST_OF_GOODS_SOLD', 'ARD_SALES_MKT_ADVERTISING_EXP',
        #     'ARD_GROSS_PROFITS', 'ARD_AMORT_EXP'
        # ]
        self._X_SELECTED_CN = ['StkIndx', 'STInt', 'dtdlevel', 'dtdtrend',
       'liqnonfinlevel', 'liqnonfintrend', 'ni2talevel', 'ni2tatrend',
       'sizelevel', 'sizetrend', 'm2b', 'sigma', 'liqfinlevel', 'lqfintrend',
       'DTDmedianFin', 'DTDmedianNonFin', 'dummySOE']

        self._Y_SELECTED = ['SALES_REV_TURN', 'CF_CASH_FROM_OPER', 'ARD_CAPITAL_EXPENDITURES', 'EBITDA']

        self._INDUSTRY = [
            'INDUSTRY_SECTOR_NUM', 'INDUSTRY_GROUP_NUM', 'INDUSTRY_SUBGROUP_NUM',
            'Industry_level_4_num', 'Industry_level_5_num', 'Industry_level_6_num'
        ]
        self._INDUSTRY_TIC_INFO = [i + '_mapped' for i in self._INDUSTRY] + ['ID_BB_UNIQUE'] + ['TICKER'] + ['Year']

    @property
    def X_SELECTED_CN(self):
        return self._X_SELECTED_CN

    @property
    def Y_SELECTED(self):
        return self._Y_SELECTED

    @property
    def INDUSTRY(self):
        return self._INDUSTRY

    @property
    def INDUSTRY_TIC_INFO(self):
        return self._INDUSTRY_TIC_INFO