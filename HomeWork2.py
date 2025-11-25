#1.  Загрузите данные и выведите информацию о них. Проверьте на наличие пропусков.

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")
df.head()
df.describe()
df.isnull().sum()

#2.   Постройте столбчатую диаграмму, сравнивающую количество здоровых и больных пациентов.

c_list = [(df["target"]==0).sum(), (df["target"]==1).sum()]
l_list = ["Здоровые", "Больные"]
plt.bar(l_list, c_list, color=(0, 1, 1))
plt.title("Сравнение количества здоровых и больных пациентов")
plt.ylabel("Количество пациентов")
plt.show()

#  3 Создайте диаграмму рассеяния, показывающую зависимость максимального пульса (thalach) от возраста (age). Раскрасьте точки в зависимости от наличия болезни.

df_healthy = df[df['target'] == 0]
df_sick = df[df['target'] == 1]

plt.figure(figsize=(10, 6))

plt.scatter(
    df_healthy['age'],
    df_healthy['thalach'],
    color='blue',
    label='Нет болезни (target=0)',
    alpha=0.6,
    edgecolors='w',
    s=60
)

plt.scatter(
    df_sick['age'],
    df_sick['thalach'],
    color='red',
    label='Есть болезнь (target=1)',
    alpha=0.6,
    edgecolors='w',
    s=60
)

plt.title('Зависимость максимального пульса от возраста, с разбивкой по наличию болезни')
plt.xlabel('Возраст (age)')
plt.ylabel('Максимальный пульс (thalach)')
plt.legend(title='Состояние')
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()

#   4. Преобразуйте признак sex (0 = женщина, 1 = мужчина) в более читаемый формат с категориями 'female' и 'male', а затем примените к нему One-Hot Encoding.

sex_map = {0: "female", 1: "male"}
df["sex"] = df["sex"].map(sex_map)
df.head()
df = pd.get_dummies(df, columns=["sex"], dtype="int")
df.head()

# 5 Рассчитайте средний уровень холестерина (chol) для больных и здоровых пациентов.

avg_chol = df.groupby('target')['chol'].mean()
print(avg_chol)

#  6 Выполните нормализацию признаков age, trestbps, chol и thalach.

# Признаки для нормализации
features = ['age', 'trestbps', 'chol', 'thalach']

for column in features:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
df.describe()

