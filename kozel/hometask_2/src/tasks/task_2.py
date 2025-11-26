from data_analyzer import DataAnalyzer
from data_finder import DataFinder
from utils import Utils

def main():

    #2. Build Sick/healthy column diagram

    df = DataFinder.get_data(Utils.DATASET_FOLDER, Utils.DATASET_FILE)
    data = DataAnalyzer(df)
    data.show_column_diagram("target", Utils.TARGET_MAPPING)

if __name__ == "__main__":
    main()
