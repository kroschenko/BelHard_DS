import pandas as pd
import matplotlib.pyplot as plt

class DataAnalyzer:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def show_column_diagram(self,
                            column: str,
                            value_mapping: dict = None) -> None:
        values = self._get_column_frame(column)

        if value_mapping:
            x_labels = [value_mapping.get(key, str(key)) for key in values.keys()]
        else:
            x_labels = [str(key) for key in values.keys()]

        y_values = list(values.values())
        plt.subplot().bar(x_labels, y_values)
        plt.ylabel("Value")

        plt.show()

    def show_scatter(self,
                     x_col: str,
                     y_col: str,
                     color_col: str,
                     color_map: dict,
                     title: str = None) -> None:

        if title is None:
            title = f'Function {y_col} from {x_col}'

        colors = [color_map[val] for val in self.df[color_col]]

        plt.scatter(
            x=self.df[x_col],
            y=self.df[y_col],
            c=colors
        )

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title)
        plt.show()

    def get_normalized_df(self, normalized_columns: list) -> pd.DataFrame:
        df = self.df.copy()
        for column in normalized_columns:
            mean_val = df[column].mean()
            min_val = df[column].min()
            max_val = df[column].max()
            range_val = max_val - min_val

            df[f'{column}_normalized'] = (self.df[column] - mean_val) / range_val
        return df

    def map_column_values(self, column: str, mapping_dict: dict) -> pd.DataFrame:
        df = self.df.copy()
        df[column] = df[column].map(mapping_dict)
        return df

    def one_hot_encode(self, column: str) -> pd.DataFrame:
        column_df = pd.get_dummies(self.df[column], dtype=int)
        df_final = pd.concat([self.df, column_df], axis=1).drop(column, axis=1)
        return df_final

    def _get_column_frame(self, column: str) -> dict:
        return self.df[column].value_counts().to_dict()
