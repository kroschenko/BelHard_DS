from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_curve, auc


class Predictor:
    def __init__(self,
                 df: pd.DataFrame,
                 model: object):
        self.df = df
        self.model = model

    def predict(self,
                target: str,
                test_size: float,
                random_state: int) -> list:
        y = self.df[target]
        x = self.df.drop(target, axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=test_size,
                                                            random_state=random_state)
        y_predicted = self.model.fit(x_train, y_train).predict(x_test)
        all_points = x_test.shape[0]
        unpredicted = (y_test != y_predicted).sum()
        return [y_test, y_predicted, all_points, unpredicted, x_test]

    @staticmethod
    def get_metrics(y_test: list,
                    y_predicted: list) -> list[float | int | Any]:
        confusion = confusion_matrix(y_test, y_predicted)
        accuracy = accuracy_score(y_test, y_predicted)
        precision = precision_score(y_test, y_predicted)
        f1 = f1_score(y_test, y_predicted)
        recall = recall_score(y_test, y_predicted)
        return [confusion, accuracy, precision, f1, recall]

    def get_roc_params(self,
                       x_test: list,
                       y_test: list) -> list:
        y_probs = self.model.predict_proba(x_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        return [fpr, tpr]

    @staticmethod
    def get_roc_auc(fpr: list,
                    tpr: list) -> float:
        return auc(fpr, tpr)

    @staticmethod
    def plot_roc(fpr: list, tpr: list,) -> None:
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {Predictor.get_roc_auc(fpr, tpr):.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Random classifier (AUC = 0.5)')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Analysis')
        plt.grid(True)
        plt.show()
