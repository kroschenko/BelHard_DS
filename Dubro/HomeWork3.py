import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import roc_curve, auc
df = pd.read_csv("heart.csv")
df.head()
df.describe()
df.isnull().sum()
y = df["target"]
X = df.drop("target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)
gnb = GaussianNB()
y_pred = gnb.fit(
    X_train, y_train
    ).predict(X_test)

print(f"Количество неправильно классифицированных точек из {X_test.shape[0]} : {(y_test != y_pred).sum()}")
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
f1_score(y_test, y_pred)
recall_score(y_test, y_pred)
y_probs = gnb.predict_proba(X_test)[:, 1]
print("Первые 5 предсказанных вероятностей положительного класса:")
print(y_probs[:5])
# Расчет FPR, TPR и порогов
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
# Расчет Площади под кривой (Area Under the Curve, AUC)
# AUC показывает общую способность модели различать классы.
# Значение 1.0 — идеальный классификатор, 0.5 — случайное угадывание.
roc_auc = auc(fpr, tpr)
print(f"\nПлощадь под ROC-кривой (AUC) для модели: {roc_auc:.4f}")

# Построение ROC-кривой
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.4f})')
# Построение линии случайного классификатора (диагональ)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
label='Случайный классификатор (AUC = 0.5)')
# Настройка осей и заголовка
plt.xlabel('False Positive Rate (FPR) / (1 - Специфичность)')
plt.ylabel('True Positive Rate (TPR) / Чувствительность')
plt.title('Receiver Operating Characteristic (ROC) Analysis')
plt.grid(True)
plt.show()

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(
	random_state=77, max_iter=1000
).fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(y_pred)

print(f"Количество неправильно классифицированных точек из {X_test.shape[0]} : {(y_test != y_pred).sum()}")
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
f1_score(y_test, y_pred)
recall_score(y_test, y_pred)
print("Первые 5 предсказанных вероятностей положительного класса:")
print(y_probs[:5])
# Расчет FPR, TPR и порогов
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
# Расчет Площади под кривой (Area Under the Curve, AUC)
# AUC показывает общую способность модели различать классы.
# Значение 1.0 — идеальный классификатор, 0.5 — случайное угадывание.
roc_auc = auc(fpr, tpr)
print(f"\nПлощадь под ROC-кривой (AUC) для модели: {roc_auc:.4f}")

# Построение ROC-кривой
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.4f})')
# Построение линии случайного классификатора (диагональ)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
label='Случайный классификатор (AUC = 0.5)')
# Настройка осей и заголовка
plt.xlabel('False Positive Rate (FPR) / (1 - Специфичность)')
plt.ylabel('True Positive Rate (TPR) / Чувствительность')
plt.title('Receiver Operating Characteristic (ROC) Analysis')
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

acc_series = []

for k in range(1, 10):
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	acc_series.append(accuracy)

plt.plot(range(1, 10), acc_series)
plt.show()

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"Количество неправильно классифицированных точек из {X_test.shape[0]} : {(y_test != y_pred).sum()}")
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
f1_score(y_test, y_pred)
recall_score(y_test, y_pred)

y_probs = gnb.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
print(f"\nПлощадь под ROC-кривой (AUC) для модели: {roc_auc:.4f}")

# Построение ROC-кривой
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.4f})')
# Построение линии случайного классификатора (диагональ)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
label='Случайный классификатор (AUC = 0.5)')
# Настройка осей и заголовка
plt.xlabel('False Positive Rate (FPR) / (1 - Специфичность)')
plt.ylabel('True Positive Rate (TPR) / Чувствительность')
plt.title('Receiver Operating Characteristic (ROC) Analysis')
plt.grid(True)
plt.show()
