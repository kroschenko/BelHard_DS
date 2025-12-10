from data_finder import DataFinder
from interactor import Interactor
from utils import Utils
from predictor import Predictor


def main():
    model = Interactor.choose_model()

    df = DataFinder.get_data("datasets", "heart.csv")

    predictor = Predictor(df, model)
    pred_data = predictor.predict("target", Utils.TEST_SIZE, Utils.RANDOM_STATE)

    print(f"Number of unpredicted targets of {pred_data[2]}: {pred_data[3]}")

    metrics = predictor.get_metrics(pred_data[0], pred_data[1])

    Interactor.print_metrics(metrics)

    roc_params = predictor.get_roc_params(pred_data[4], pred_data[0])
    print(f"ROC AUC: {predictor.get_roc_auc(roc_params[0], roc_params[1])}")

    predictor.plot_roc(roc_params[0], roc_params[1])


if __name__ == "__main__":
    main()
