# Star Type Classification API (AstroVision)

FastAPI-застосунок: тренування моделі RandomForest та інференс типу зорі. Усі дії виконуються через HTTP-запити,
а результати логуються в базу даних `astro_vision.db` (SQLite).

## Структура проєкту

```
lab4_project/
├── app/
│   ├── main.py                 # FastAPI-застосунок, підключення роутерів, створення таблиць при старті
│   ├── schemas.py              # Pydantic-схеми запитів/відповідей
│   ├── database/               # підключення до БД (database.py) та ORM-моделі (models.py)
│   ├── endpoints/              # training.py (/train-model), inference.py (/predict, /predictions)
│   └── services/               # model.py (тренування/передбачення), utils.py (метадані моделі)
├── models/                     # best_star_classifier_rf.pkl, model_meta.json, feature_importances.csv
├── logs/training.log           # лог тренування
├── src/                        # ноутбуки та скрипти 
├── astro_vision.db             # база даних SQLite
├── stars.csv, stars_with_timestamp.csv
└── requirements.txt
```

## Запуск сервера

```bash
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --reload
```

Документація: Swagger UI — http://127.0.0.1:8000/docs, ReDoc — http://127.0.0.1:8000/redoc

## Тренування моделі — `POST /train-model`

```bash
curl -X POST http://127.0.0.1:8000/train-model
```

Що відбувається: читаються всі записи з `stars_data_cleaned` - стратифікований поділ 90:10 на `train` та
`new_input` - тренування RandomForest (гіперпараметри з ЛР3, `class_weight="balanced"`) - збереження моделі в `models/` - запис передбачень на train у `predictions` (`source="train"`) - збереження 10 % у таблицю `new_input`.

Приклад відповіді:

```json
{
  "message": "Модель натреновано та збережено.",
  "n_train": 9000,
  "n_new_input": 1000,
  "train_accuracy": 0.9974,
  "new_input_accuracy": 0.985,
  "new_input_f1_weighted": 0.9852,
  "model_path": ".../models/best_star_classifier_rf.pkl",
  "train_predictions_logged": 9000
}
```

## Передбачення — `POST /predict`

Варіант 1: за `star_id` (ознаки беруться з `stars_data_cleaned`):

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"star_id": "STAR_06302"}'
```

Варіант 2: JSON з усіма 16 ознаками (значення в тому вигляді, в якому вони зберігаються в `stars_data_cleaned`,
тобто після кодування й масштабування):

```json
{
  "Temperature_K": -0.7133,
  "Luminosity_L_Sun": -0.6383,
  "Radius_R_Sun": -0.7912,
  "Mass_M_Sun": -0.8334,
  "Distance_LY": -0.6896,
  "Apparent_Magnitude": -0.2194,
  "Absolute_Magnitude": 0.6308,
  "Spectral_Class": 5,
  "Star_Color": 3,
  "Age_Million_Years": 1.407,
  "Metallicity": -0.7048,
  "Rotation_Speed_KM_S": 0.1405,
  "Constellation": 11,
  "Galaxy_Type": 0,
  "Variable_Star": 1,
  "Surface_Gravity": -0.5514
}
```

Відповідь:

```json
{"prediction_id": 49003, "input_id": 1, "predicted_type": 3, "predicted_label": "Main Sequence",
 "source": "inference", "probability": 1.0}
```

Помилки: порожній запит або неповний набір ознак — `422`; невідомий `star_id` — `404`; модель ще не натренована — `409`.
Історія передбачень: `GET /predictions?limit=20&source=inference`.

## Структура БД

| Таблиця | Призначення |
|---|---|
| `stars`, `stars_clean` | вихідні дані (ЛР1) |
| `stars_data_cleaned` | очищені/закодовані/масштабовані ознаки + `Star_Type` + `timestamp` , зв'язок 1:1 зі `stars` |
| `new_input` | 10 % записів, відкладених під час `/train-model` |
| `predictions` | `id, Star_ID, model_name, true_label, prediction, source ("train"/"inference"), timestamp, input_id` |
| `inference_inputs` | `id, star_id, 16 ознак, timestamp` — вхідні параметри кожного запиту `/predict` |
| `model_metrics` | метрики моделей з ЛР3 |

## Приклади збережених передбачень

| id | Star_ID | model_name | true_label | prediction | source | timestamp | input_id |
|---|---|---|---|---|---|---|---|
| 49003 | STAR_06302 | RandomForest_API | 3 | 3 | inference | 2026-11-03 20:02:15 | 1 |
| 49004 | — | RandomForest_API | — | 4 | inference | 2026-11-03 20:02:15 | 2 |
| 49005 | — | RandomForest_API | — | 3 | inference | 2026-11-03 20:02:15 | 3 |
