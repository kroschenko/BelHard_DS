import pandas as pd
from pathlib import Path

class DataFinder:

    @staticmethod
    def get_data(folder: str, file: str) -> pd.DataFrame:
        script_dir = Path(__file__).parent
        csv_path = script_dir.parent.parent.parent.parent / folder / file
        return pd.read_csv(csv_path)
