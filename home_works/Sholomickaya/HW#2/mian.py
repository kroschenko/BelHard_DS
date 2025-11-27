# Выборка Heart Disease. Содержит медицинские данные пациентов,
# такие как возраст, пол, уровень холестерина, и наличие заболевания
# сердца.
# 1. Загрузите данные и выведите информацию о них. Проверьте на
# наличие пропусков.
# 2. Постройте столбчатую диаграмму, сравнивающую количество
# здоровых и больных пациентов.
# 3. Создайте диаграмму рассеяния, показывающую зависимость
# максимального пульса (thalach) от возраста (age). Раскрасьте точки в
# зависимости от наличия болезни.
# 4. Преобразуйте признак sex (0 = женщина, 1 = мужчина) в более
# читаемый формат с категориями 'female' и 'male', а затем примените к
# нему One-Hot Encoding.
# 5. Рассчитайте средний уровень холестерина (chol) для больных и
# здоровых пациентов.
# 6. Выполните нормализацию признаков age, trestbps, chol и thalach.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataAnalayzer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)

    def show_info(self):
        """1. Показывает информацию о данных"""
        print("=== ИНФОРМАЦИЯ О ДАННЫХ ===")
        print(f"Размер данных: {self.data.shape}")
        print("\nПервые 5 строк:")
        print(self.data.head())
        print("\nИнформация о типах данных:")
        print(self.data.info())
        print("\nПропущенные значения:")
        print(self.data.isnull().sum())

    def plot_health_distribution(self):
        """2. Строит столбчатую диаграмму сравнивающую количество здоровых и больных пациентов"""

        # Считаем количество здоровых и больных
        health_counts = self.data['target'].value_counts()

        print("Распределение пациентов:")
        print(health_counts)

        # Подписи для столбцов 1 - больные, 0 - здоровые
        labels = ['Больные', 'Здоровые']

        # Диаграмма
        plt.figure(figsize=(8, 6))
        plt.bar(labels, health_counts.values, color=['red', 'green'])
        plt.title('Количество здоровых и больных пациентов')
        plt.ylabel('Количество пациентов')

        # Добавляем числа над столбцами
        for i, count in enumerate(health_counts.values):
            plt.text(i, count + 5, str(count), ha='center')

        plt.show()

    def plot_pulse_vs_age(self):
        """3 Диаграмма рассеяния: пульс vs возраст, раскрашенная по наличию болезни"""

        plt.figure(figsize=(10, 6))

        # Разделяем данные на две группы
        sick_patients = self.data[self.data['target'] == 1]  # больные
        healthy_patients = self.data[self.data['target'] == 0]  # здоровые

        # Точки для каждой группы
        plt.scatter(sick_patients['age'], sick_patients['thalach'],
                    alpha=0.7, label='Больные', color='red', s=50)
        plt.scatter(healthy_patients['age'], healthy_patients['thalach'],
                    alpha=0.7, label='Здоровые', color='green', s=50)

        plt.xlabel('Возраст (age)')
        plt.ylabel('Максимальный пульс (thalach)')
        plt.title('Зависимость максимального пульса от возраста')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def transform_sex_with_dummies(self):
        """4. Преобразование признака sex (0 = женщина, 1 = мужчина) в более
            читаемый формат с категориями 'female' и 'male', а затем применение к нему One-Hot Encoding. """

        print("ИСХОДНЫЕ ДАННЫЕ:")
        print(self.data['sex'].head())
        print(f"\nУникальные значения: {self.data['sex'].unique()}")

        print("\n" + "=" * 40)
        print("ПРЕОБРАЗОВАНИЕ В ЧИТАЕМЫЙ ФОРМАТ:")
        sex_mapping = {0: 'female', 1: 'male'}
        self.data['sex_text'] = self.data['sex'].map(sex_mapping)

        print("Результат преобразования (первые 5 строк):")
        print(self.data[['sex', 'sex_text']].head())

        sex_dummies = pd.get_dummies(self.data['sex_text'], prefix='sex')


        # axis=1 - по столбцам
        self.data = pd.concat([self.data, sex_dummies], axis=1)

        print("Итоговые данные (первые 5 строк):")
        # Показываем только relevant столбцы
        result_columns = ['sex', 'sex_text', 'sex_female', 'sex_male']
        print(self.data[result_columns].head())

    def calculate_avg_cholesterol_simple(self):
        """5. Рассчитывает средний уровень холестерина (chol) для больных и здоровых пациентов."""

        result = self.data.groupby('target')['chol'].mean()

        print(f"=" * 40)
        print("Средний уровень холестерина:")
        print(f"Для здоровых (target=0): {result[0]:.2f} мг/дл")
        print(f"Для больные (target=1): {result[1]:.2f} мг/дл")


    def normalize_with_sklearn(self):
        """6. Нормализация признаков age, trestbps, chol и thalach"""

        features = ['age', 'trestbps', 'chol', 'thalach']

        # Нормализация
        normalized_values = MinMaxScaler().fit_transform(self.data[features])

        normalized_df = pd.DataFrame(normalized_values, columns=[f'{feat}_normalized' for feat in features])
        self.data = pd.concat([self.data, normalized_df], axis=1)

        print(f"=" * 40)
        print(f"Нормализация завершена")
        print(self.data.head())





heart = DataAnalayzer("../../datasets/heart.csv")
heart.show_info()
heart.plot_health_distribution()
heart.plot_pulse_vs_age()
heart.transform_sex_with_dummies()
heart.calculate_avg_cholesterol_simple()
heart.normalize_with_sklearn()
