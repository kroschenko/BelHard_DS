import pandas as pd

from data_analyzer import DataAnalyzer
from data_finder import DataFinder
from utils import Utils

def main():

    #6. Get normalization for columns

    df = DataFinder.get_data(Utils.DATASET_FOLDER, Utils.DATASET_FILE)
    data = DataAnalyzer(df)

    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 400)

    print(data.get_normalized_df(Utils.NORMALIZATION_COLUMNS))

if __name__ == "__main__":
    main()
