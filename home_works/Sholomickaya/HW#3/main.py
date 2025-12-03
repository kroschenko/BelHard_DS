import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# 1 Взять за основу датасет Heart Disease UCI из ДЗ 2
df = pd.read_csv("heart.csv")

# 2 Загрузить данные, разделить их на обучающую и тестовую выборки + масштабирование
X = df.drop('target', axis=1) # матрица признаков
y = df['target'] # целевая переменная

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # масштабируем
X_test_scaled = scaler.transform(X_test)

# 3 Обучить модели наивного байесовского классификатора,
# логистической регрессии и k-ближайших соседей
# (выявить k с наилучшим результатом, например, путем перебора результатов,
# получаемых для классификаторов с разными значениями k)

# Наивный Байес
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)

# Логистическая регрессия
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# KNN с подбором лучшего k
best_k = 1
best_acc = 0

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    if acc > best_acc:
        best_acc = acc
        best_k = k

# Финальная модель KNN
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)
y_pred_knn = knn_best.predict(X_test_scaled)

print("Результаты обучения моделей:")
print(f"Наивный Байес - Accuracy: {accuracy_score(y_test, y_pred_nb):.3f}")
print(f"Логистическая регрессия - Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}")
print(f"KNN (k={best_k}) - Accuracy: {best_acc:.3f}")


# 4 Построить матрицу ошибок, оценить модель с помощью accuracy, precision, recall и F1-score
models = {
    'Naive Bayes': (y_pred_nb, 'Наивный Байес'),
    'Logistic Regression': (y_pred_lr, 'Логистическая регрессия'),
    f'KNN (k={best_k})': (y_pred_knn, f'KNN с k={best_k}')
}

# Оцениваем каждую модель
for model_name, (y_pred, model_label) in models.items():
    print(f"\n{'=' * 60}")
    print(f"Модель: {model_label}")
    print('=' * 60)

    #  Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    print("Матрица ошибок (Confusion Matrix):")
    print(f"           Предсказано 0  Предсказано 1")
    print(f"Реально 0:     {cm[0, 0]:4}          {cm[0, 1]:4}")
    print(f"Реально 1:     {cm[1, 0]:4}          {cm[1, 1]:4}")

    # Основные метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nМетрики:")
    print(f"Accuracy  (Точность):          {accuracy:.3f}")
    print(f"Precision (Точность/Precision): {precision:.3f}")
    print(f"Recall    (Полнота/Recall):    {recall:.3f}")
    print(f"F1-Score  (F-мера):            {f1:.3f}")

    #  Отчет
    print(f"\nПодробный отчет:")
    print(classification_report(y_test, y_pred, target_names=['Здоров (0)', 'Болен (1)']))

    # 5 Провести ROC-анализ обученных классификаторов

    y_proba_nb = nb.predict_proba(X_test_scaled)[:, 1]
    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    y_proba_knn = knn_best.predict_proba(X_test_scaled)[:, 1]

    # Рассчитываем ROC-кривые для каждой модели
    fpr_nb, tpr_nb, _ = roc_curve(y_test, y_proba_nb)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)

    # Рассчитываем AUC (Area Under Curve)
    auc_nb = roc_auc_score(y_test, y_proba_nb)
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    auc_knn = roc_auc_score(y_test, y_proba_knn)

    # Строим ROC-кривые
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {auc_nb:.3f})', linewidth=2)
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})', linewidth=2)
    plt.plot(fpr_knn, tpr_knn, label=f'KNN k={best_k} (AUC = {auc_knn:.3f})', linewidth=2)

    # Диагональ (случайный классификатор)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)', alpha=0.5)

    # Настройки графика
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR) / Recall', fontsize=12)
    plt.title('ROC-кривые для разных моделей', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Добавляем сетку и подписи
    plt.fill_between(fpr_lr, tpr_lr, alpha=0.1)
    plt.fill_between(fpr_nb, tpr_nb, alpha=0.1)
    plt.fill_between(fpr_knn, tpr_knn, alpha=0.1)

    plt.tight_layout()
    plt.show()

    # Вывод AUC значений
    print("\n" + "=" * 60)
    print("ROC-AUC значения:")
    print("=" * 60)
    print(f"Наивный Байес:          AUC = {auc_nb:.4f}")
    print(f"Логистическая регрессия: AUC = {auc_lr:.4f}")
    print(f"KNN (k={best_k}):        AUC = {auc_knn:.4f}")

    # Создаем итоговую таблицу сравнения
    print("\n" + "=" * 70)
    print("ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ")
    print("=" * 70)

    results = []
    for model_name, (y_pred, model_label) in models.items():
        y_proba = None
        if model_name == 'Naive Bayes':
            y_proba = y_proba_nb
        elif model_name == 'Logistic Regression':
            y_proba = y_proba_lr
        else:
            y_proba = y_proba_knn

        results.append({
            'Модель': model_label,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_proba)
        })

    # Создаем DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AUC', ascending=False)
    print(results_df.to_string(index=False))
