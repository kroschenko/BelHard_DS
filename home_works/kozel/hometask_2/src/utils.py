class Utils:

    DATASET_FOLDER = "datasets"
    DATASET_FILE = "heart.csv"

    TARGET_MAPPING = {
        1 : "Sick",
        0 : "Healthy"
    }

    COLOR_MAP = {
        1: "red",
        0: "green"
    }

    SEX_MAPPING = {
        1 : "male",
        0 : "female"
    }

    NORMALIZATION_COLUMNS = ["age", "trestbps", "chol", "thalach"]
