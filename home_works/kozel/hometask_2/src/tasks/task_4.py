import pandas as pd

from data_analyzer import DataAnalyzer
from data_finder import DataFinder
from utils import Utils

def main():

    # 4. Modify data. Using One-Hot Encoding

    #4.1 Modify 'sex' column

    original_df = DataFinder.get_data(Utils.DATASET_FOLDER, Utils.DATASET_FILE)
    data = DataAnalyzer(original_df)
    df = data.map_column_values("sex", Utils.SEX_MAPPING)
    print(f"\nDataframe with 'sex' column modified:\n{df}")

    #4.2 One-Hot Encoding

    data.df = df
    df_final = data.one_hot_encode("sex")
    print(f"\nDataframe with One-Hot Encoding:\n{df_final}")

if __name__ == "__main__":
    main()
