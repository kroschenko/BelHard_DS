from data_analyzer import DataAnalyzer
from data_finder import DataFinder
from utils import Utils

def main():

    # 3. Build age/thalach scatter

    df = DataFinder.get_data(Utils.DATASET_FOLDER, Utils.DATASET_FILE)
    data = DataAnalyzer(df)
    data.show_scatter("age", "thalach", "target", Utils.COLOR_MAP)

if __name__ == "__main__":
    main()
