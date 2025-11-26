from data_finder import DataFinder
from utils import Utils

def main():

    #1. Describe data and analysis it on NaN

    df = DataFinder.get_data(Utils.DATASET_FOLDER, Utils.DATASET_FILE)
    print(f"{df.describe()}\n")
    print(f"NaN values:\n{df.isnull().sum()}")

if __name__ == "__main__":
    main()
