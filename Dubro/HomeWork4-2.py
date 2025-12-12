import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random as rnd
import math

class KohonenMap:
    def __init__(self, shape, dimension, rate0, sigma0, tau2):
        # shape теперь ожидается как кортеж (N, M)
        self.weights = np.random.random((shape[0], shape[1], dimension))
        self.tau2 = tau2
        self.rate0 = rate0
        self.rate = rate0
        self.sigma0 = sigma0
        self.sigma = sigma0
        self.shape = shape  # Храним кортеж

    def core(self, data, iterationsLimit, changeRate):
        samples_count = len(data)
        iterations = 0
        while iterations < iterationsLimit:
            index_rnd = rnd.randint(0, samples_count - 1)
            # data теперь массив NumPy, используем стандартное индексирование
            sample = data[index_rnd]

            win_index = self._define_win_neuron(sample)
            topological_locality = self._topological_locality(win_index)

            for i in range(0, self.shape[0]):
                for j in range(0, self.shape[1]):
                    self.weights[i, j] += self.rate * topological_locality[i, j] * (sample - self.weights[i, j])

            iterations += 1
            if changeRate:
                self._change_rate(iterations)
            self._change_sigma(iterations)

    def train(self, data):
        self.core(data, 1000, True)
        self.rate = 0.01
        self.core(data, 25000, False)

    def print_clusters(self, data):
        clustering = []
        for sample in data:
            index = self._define_win_neuron(sample)
            clustering.append(index[0] + index[1])
        return clustering

    def _define_win_neuron(self, sample):
        dist = 1e6
        row = col = -1
        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                if np.linalg.norm(sample - self.weights[i, j]) < dist:
                    dist = np.linalg.norm(sample - self.weights[i, j])
                    row = i
                    col = j
        return [row, col]

    def _topological_locality(self, index):
        distance = np.zeros((self.shape[0], self.shape[1]))
        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                distance[i, j] = np.linalg.norm(index - np.array([i, j])) ** 2
        return np.exp(-distance / (2 * self.sigma ** 2))

    def _change_sigma(self, n):
        log_sigma0 = math.log(self.sigma0) if self.sigma0 > 1 else math.log(2.0)
        tau1 = 1000.0 / log_sigma0
        self.sigma = self.sigma0 * math.exp(-n / tau1)

    def _change_rate(self, n):
        self.rate = self.rate0 * math.exp(-n / self.tau2)

# 1. Загрузка данных
df = pd.read_csv("Seed_Data.csv")

# 2. Подготовка данных: отделяем признаки от целевой переменной
def prepareData(dataset):
    # ИСПРАВЛЕНИЕ: Преобразуем DataFrame в NumPy массивы сразу
    return dataset.drop("target", axis=1).to_numpy(), dataset["target"].to_numpy()

data, target_actual = prepareData(df)

# 3. Кластеризация KMeans
N_CLUSTERS = 3
kmeans_model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels_predicted = kmeans_model.fit_predict(data)

# 4. Снижение размерности с помощью PCA для визуализации
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(data)


# 5. Функция для построения графика
def plot_data_with_labels(X_reduced_data, labels, title="KMeans Clustering Results"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced_data[:, 0], X_reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.7,
                          edgecolors='w', s=100)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.title(title)
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.grid(True)

# 6. Визуализация результатов (используем предсказанные метки)
plot_data_with_labels(X_reduced, labels_predicted, title="KMeans Predicted Clusters")

# Задаем параметры для сети Кохонена
SOM_SHAPE = (5, 5)  # Используем кортеж (rows, cols)
DIMENSION = data.shape[1]  # Количество признаков

# Создание экземпляра класса KohonenMap
som_map = KohonenMap(SOM_SHAPE, DIMENSION, 0.1, 2.0, 1000.0)

# Обучение карты
som_map.train(data)

# Получение меток кластеров
som_labels = som_map.print_clusters(data)

# Расчет и вывод силуэтного коэффициента для KohonenMap
silhouette_avg_som = silhouette_score(data, som_labels)
print(f"Силуэтный коэффициент (KohonenMap): {silhouette_avg_som:.4f}")

# Также выводим силуэт для K-Means для сравнения
silhouette_avg_kmeans = silhouette_score(data, labels_predicted)
print(f"Силуэтный коэффициент (KMeans): {silhouette_avg_kmeans:.4f}")

plt.show()