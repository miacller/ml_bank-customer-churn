# Churn Prediction Project

## Описание проекта
Проект посвящён предсказанию ухода клиентов (churn) для телеком/финансовых сервисов. Цель - выявлять клиентов с высоким риском оттока, чтобы своевременно принимать меры по удержанию.

В проекте рассматриваются три модели:

1. **Logistic Regression** - baseline.
2. **Random Forest** - улучшение метрик за счёт ансамбля деревьев, оптимизация порога классификации.
3. **LightGBM** - градиентный бустинг, оптимизация F2-score и порога классификации.

Проект демонстрирует:
- Подготовку и предобработку данных
- Построение базовых моделей
- Оптимизацию гиперпараметров
- Оптимизацию порога классификации для бизнес-метрик
- Визуализацию метрик (ROC, Precision-Recall, F2 vs Threshold)
- Сравнение моделей

---

## Данные
- Исходный датасет (`Customer-Churn-Records.csv`) содержит информацию о клиентах: 
  - возраст, пол, активность, подписки, услуги и прочее
  - целевая переменная `Exited` - ушёл (1) / остался (0)
- Размер: 10 000 строк × 15 признаков
- Для обучения моделей используется **предобработанный датасет** `processed_data.csv`

*Датасет "Bank Customer Churn Prediction" доступен на Kaggle: https://www.kaggle.com/datasets*
---

## Установка и запуск
1. Установить зависимости:

pip install -r requirements.txt

2. Запустить Jupyter Notebook:

jupyter notebook

3. Последовательно выполнить ноутбуки:

01_baseline_logistic_regression.ipynb

02_random_forest.ipynb

03_lightgbm.ipynb

---

Результаты
Итоговые метрики (тестовая выборка)
| Модель | Accuracy | Precision | Recall | F2-score | ROC-AUC |
|------|------|------|------|------|------|
| Logistic Regression | 0.525 | 0.247 | 0.654 | 0.492 | 0.601 |
| Random Forest | **0.846** | 0.388 | 0.843 | 0.685 | 0.860 |
| **LightGBM** | 0.738 | **0.429** | **0.855** | **0.713** | **0.874** |

Вывод: LightGBM показывает лучший F2-score и Recall, что критично для задач удержания клиентов.

## 🔹 Бизнес-ценность
- Модель позволяет **обнаружить до 85% уходящих клиентов**  
- Выбор **оптимального порога** помогает настроить **баланс между Precision и Recall**  
- Решение можно интегрировать в систему **Retention Marketing**, чтобы **снизить отток и увеличить доход**

---

## Визуализации:
- ROC-кривые для всех моделей
- Матрицы корреляций и признаков
- Feature importance
- Precision-Recall кривая с отмеченным оптимальным порогом
- График F2-score vs Threshold для выбора бизнес-порога

---

## Структура проекта
project/
│
├─ data/ - подготовленный датасет
│ └─ processed_data.csv
│ └─ Customer-Churn-Records.csv
├─ notebooks/
│ ├─ 01_baseline_logistic_regression.ipynb
│ ├─ 02_random_forest.ipynb
│ └─ 03_lightgbm.ipynb
├─ models/ - сохранённые модели и оптимальные пороги
│ ├─ random_forest_final_model.pkl
│ └─ lightgbm_final_model.txt
├─ requirements.txt - зависимости 
└─ README.md - описание проекта

---

## Возможные улучшения:
- Feature engineering: новые признаки, взаимодействия, агрегаты
- Работа с дисбалансом классов (SMOTE, undersampling)
- Более сложные модели: CatBoost, XGBoost
- Stacking/ensembling моделей

---

## Технологии:
- Python 3.11
- jupyter notebook
- pandas, numpy
- scikit-learn
- LightGBM
- matplotlib, seaborn
- joblib

---

## Автор
Tishchenko Kirill
Junior/Middle Data Scientist
GitHub: https://github.com/miacller
LinkedIn: in/kirill-tishchenko
