#1
import pandas as pd

pd.set_option("display.width", 10000) # огромная ширина строки
pd.set_option("display.max_columns", None) # показывать все колонки
pd.set_option("display.max_colwidth", None) # не обрезать текст

df = pd.read_csv("heart.csv")

print(df)

print(f"""
Количество пропусков: 
{df.isnull().sum()}""")

#2
import matplotlib.pyplot as plt

counts = df["target"].value_counts()
condition = counts.rename({1: "Больные", 0: "Здоровые"}) # переименовываем индексы для наглядности

print("Данные о состоянии пациентов:")
for key, value in condition.items():
    print(f"{key}: {value}")


fig, ax = plt.subplots(figsize=(8, 6)) # figsize - задаем рамер графика в дюймах

patients = ["Здоровые", "Больные"]
counts = df["target"].value_counts().sort_index() # считаем количество здоровых (0) и больных (1) пациентов
bar_colors = ['tab:green', 'tab:red']

ax.bar(patients, counts.values, label = patients, color=bar_colors) # label = patients - для легенды нужны одинаковые значения
ax.set_ylabel("Количество пациентов")
ax.set_xlabel("Состояние пациента")
ax.set_title("Сравнение здоровых и больных пациентов")
ax.legend(loc="upper left") # loc - задаем месторасположение легенды

plt.show()

#3
dots_colors = df["target"].map({0: "green", 1: "red"}) # Создаем цвета для точек

# Рисуем диаграмму рассеяния
plt.figure(figsize=(8, 6))
plt.scatter(df["age"], df["thalach"], color=dots_colors)

plt.xlabel("Возраст, лет")
plt.ylabel("Максимальный пульс, уд/мин")
plt.title("Максимальный пульс здоровых и больных пациентов (по возрасту)")

plt.show()

#4
sex_mapping = {0:"female", 1:"male"}

df_sex = df.copy() # делаем копию исходной таблицы, чтобы не менять исходные данные

df_sex["sex"] = df_sex["sex"].map(sex_mapping) # map - заменяет собой указанные строковые значения в соответствующем столбце
print(df_sex)

sex_dummies = pd.get_dummies(df_sex["sex"], dtype = int) # применяем one-hot encoding
print(pd.concat([df.drop(columns="sex"), sex_dummies], axis=1)) # конкатенируем столбцы

#5
chol_mean = df.groupby("target")["chol"].mean() # groupby - разделяет датасет на группы в зависимости от данных (по значению индексов),  далее цепочка - выбор колонки внутри группы (["chol"]), а mean - считаем среднее значение для каждой группы

print(f"""Средний уровень холестерина:
Здоровые пациенты: {chol_mean[0]:.2f}
Больные  пациенты: {chol_mean[1]:.2f}""")

#6
df_min_max = df.copy() # делаем копию исходной таблицы, чтобы не менять исходные данные
columns = ["age", "trestbps", "chol", "thalach"]

for c in columns:
    df_min_max[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min()) #минимаксная нормализация

print(f"""Минимаксная нормализация (age, trestbps, chol, talach):
{df_min_max}""")


df_avr = df.copy() # делаем копию исходной таблицы, чтобы не менять исходные данные
columns = ["age", "trestbps", "chol", "thalach"]

for c in columns:
    df_avr[c] = (df[c] - df[c].mean()) / df[c].std() # нормализация средним

print(f"""Нормализация средним (age, trestbps, chol, talach):
{df_min_max}""")

