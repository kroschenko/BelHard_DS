from data_analyzer import DataAnalyzer
from data_finder import DataFinder
from utils import Utils

def main():

    #5. Finding average chol for sick and healthy people

    df_mapped = DataFinder.get_data(Utils.DATASET_FOLDER, Utils.DATASET_FILE)
    data = DataAnalyzer(df_mapped)
    df_mapped = data.map_column_values("target", Utils.TARGET_MAPPING)
    df_mapped = df_mapped.groupby("target")["chol"].mean()
    print(df_mapped)

if __name__ == "__main__":
    main()
