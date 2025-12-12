import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

column_names = [
    "X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"
]

raw_dataset = pd.read_csv("forestfires.csv")

raw_dataset.isna().sum()
dataset = raw_dataset.dropna()

key_features_for_plot = ['temp', 'RH', 'wind', 'rain', 'area']

sns.pairplot(
    dataset[key_features_for_plot],
    diag_kind='kde'
)
plt.show()

dataset = pd.get_dummies(
    dataset,
    columns=['month', 'day'],
    drop_first=True,
    dtype=int
)

dataset['log_area'] = np.log1p(dataset['area'])

features = dataset.drop(['area', 'log_area'], axis=1)
target = dataset['log_area']

train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    features, target, test_size=0.2, random_state=0
)

train_dataset.describe().transpose()

scaler = StandardScaler()
train_dataset_scaled = scaler.fit_transform(train_dataset)
test_dataset_scaled = scaler.transform(test_dataset)

metrics = {}

reg = LinearRegression()
reg.fit(train_dataset_scaled, train_labels)

y_test_linear_predict = reg.predict(test_dataset_scaled)

metrics['Linear Model'] = {
    'MSE': mean_squared_error(y_test_linear_predict, test_labels),
    'MAE': mean_absolute_error(y_test_linear_predict, test_labels),
    'R^2': r2_score(y_test_linear_predict, test_labels)
}

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(train_dataset_scaled)
X_test_poly = poly.transform(test_dataset_scaled)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, train_labels)

y_test_polynomial_predict = model_poly.predict(X_test_poly)

metrics['PolynomialModel'] = {
    'MSE': mean_squared_error(y_test_polynomial_predict, test_labels),
    'MAE': mean_absolute_error(y_test_polynomial_predict, test_labels),
    'R^2': r2_score(y_test_polynomial_predict, test_labels)
}

results_df = pd.DataFrame(metrics).T
print(results_df.round(4))

def plot_predictions_vs_true(true_labels, predictions, title):
    plt.figure(figsize=(7, 7))
    plt.scatter(true_labels, predictions, alpha=0.6)
    plt.title(title)
    plt.xlabel("Истинные значения (log_area)")
    plt.ylabel("Предсказанные значения (log_area)")
    plt.grid(True)

    limits = [min(true_labels.min(), predictions.min()),
              max(true_labels.max(), predictions.max())]
    plt.plot(limits, limits, color='red', linestyle='--', linewidth=2)
    plt.show()


plot_predictions_vs_true(
    test_labels,
    y_test_polynomial_predict,
    "Истинные vs Предсказанные значения log(1+area) (Полиномиальная модель)"
)