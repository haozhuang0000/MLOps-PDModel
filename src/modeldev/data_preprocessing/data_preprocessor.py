import pandas as pd

class DataPreprocessor:

    def __init__(self, path: str):

        self.path = path

    def load_data(self):

        return pd.read_csv(self.path)

    def filling_data(self, df):
        col_fill_zero = ['liqfinlevel', 'lqfintrend', 'DTDmedianFin']
        df[col_fill_zero] = df[col_fill_zero].fillna(0)

        cols_to_fill = [col for col in df.columns if col != "Y"]
        df.loc[df["Y"] == 1, cols_to_fill] = df.loc[df["Y"] == 1, cols_to_fill].fillna(0)
        return df

    def preprocess_data(self):

        df = self.load_data()
        df = self.filling_data(df)
        df = df.dropna()
        df = df[df.Y!=100]
        return df

if __name__ == "__main__":
    data_preprocessor = DataPreprocessor()
    data_preprocessor.preprocess_data()
