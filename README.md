# CommonLit Readability – ML R&D

Учебный R&D-проект курса «Исследования и разработка в ML» (ИТМО, 2026) по задаче
[CommonLit Readability Prize](https://www.kaggle.com/competitions/commonlitreadabilityprize):
по тексту англоязычного отрывка (около 175 слов) предсказать легкость его чтения для школьников 3–12 классов.
Тип задачи – регрессия, основная метрика – RMSE на 5-fold CV x 3 сида с фиксированными фолдами.

## Домашние задания

| ДЗ | Тема | Результат | Отчет |
|---|---|---|---|
| 1 | Постановка задачи, EDA, бейзлайн, протокол оценки | TF-IDF + Ridge: RMSE **0.725 ± 0.015** (константа – 1.033) | [`reports/tfidf_ridge/report.md`](reports/tfidf_ridge/report.md) |
| 2 | Улучшения бейзлайна, трекинг экспериментов (MLflow), абляции | вложенный подбор alpha + 12 readability-признаков: RMSE **0.675 ± 0.015**, 7 абляций | [`reports/baseline_improvement_experiments/report.md`](reports/baseline_improvement_experiments/report.md) |
| 3 | Анализ ошибок, диагностика по срезам, проверки устойчивости, ограничения | 5 типов ошибок, 4 среза, 3 проверки устойчивости; главные слабости – края шкалы, опечатки, новые источники | [`reports/error_analysis_robustness/report.md`](reports/error_analysis_robustness/report.md) |

Состояние репозитория на момент сдачи ДЗ1 зафиксировано тегом [`hw1`](https://github.com/ArtemCh101/R-D-ML_ITMO_2026/tree/hw1).

## Основная гипотеза проекта

Дообучение предобученного трансформера roberta-base с регрессионной головой снизит RMSE относительно бейзлайна (0.725) не менее чем на 0.1, то есть до 0.625 и ниже, на тех же фолдах. Улучшение должно быть значимым по парному bootstrap и сохраняться на срезе современных текстов. Порог зафиксирован в ДЗ1 до экспериментов; подробности – в [отчете ДЗ1](reports/tfidf_ridge/report.md#основная-гипотеза).

## Установка

Python 3.11:

```bash
conda create -n rnd-hw1 python=3.11 -y
conda activate rnd-hw1
pip install -r requirements.txt
```

Данные: скачать `train.csv`, `test.csv`, `sample_submission.csv` со страницы [соревнования](https://www.kaggle.com/competitions/commonlitreadabilityprize/data) в `data/`:

```bash
kaggle competitions download -c commonlitreadabilityprize -p data
unzip data/commonlitreadabilityprize.zip -d data
```

## Запуск

Все команды выполняются из корня репозитория.

**ДЗ1 – бейзлайн** (фолды, обучение по CV, оценка; результаты в `reports/tfidf_ridge/`):

```bash
bash scripts/run_baseline.sh
```

**ДЗ2 – эксперименты** (проверка инфраструктуры, затем 10 экспериментов и сравнение; результаты в `reports/baseline_improvement_experiments/`):

```bash
bash scripts/check_infra.sh
bash scripts/run_experiments.sh
```

**ДЗ3 – анализ ошибок и устойчивости** (использует результаты ДЗ2; результаты в `reports/error_analysis_robustness/`):

```bash
bash scripts/run_analysis.sh
```

**MLflow UI** (запуски ДЗ2, эксперимент `commonlit-readability`):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**EDA:** [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb), перед запуском нужны данные и фолды (`python -m src.data`).

## Структура репозитория

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml                       # ДЗ1: бейзлайн
│   ├── experiments/                        # ДЗ2: общие правила (common.yaml) и 10 экспериментов
│   └── analysis/hw3.yaml                   # ДЗ3: правила анализа ошибок и проверок устойчивости
├── data/                                   # не в git: данные, фолды, кэш признаков
├── notebooks/
│   └── 01_eda.ipynb                        # EDA
├── reports/
│   ├── figures/                            # ДЗ1: графики EDA
│   ├── tfidf_ridge/                        # ДЗ1: отчет и результаты бейзлайна
│   ├── baseline_improvement_experiments/   # ДЗ2: отчет, графики, таблицы, OOF-предсказания
│   └── error_analysis_robustness/          # ДЗ3: отчет, графики, таблицы
├── scripts/
│   ├── run_baseline.sh                     # ДЗ1
│   ├── check_infra.sh                      # ДЗ2: проверка реестра, MLflow, сравнения
│   ├── run_experiments.sh                  # ДЗ2
│   └── run_analysis.sh                     # ДЗ3
└── src/
    ├── data.py                             # данные, фиксированные фолды
    ├── features.py                         # readability-признаки
    ├── models.py                           # бейзлайн и модели экспериментов
    ├── train.py                            # обучение по CV, OOF
    ├── evaluate.py                         # оценка бейзлайна ДЗ1
    ├── registry.py                         # реестр и проверка конфигов экспериментов
    ├── compare.py                          # сравнение моделей, парный bootstrap, графики
    ├── tracking.py                         # MLflow, git commit, хэши данных
    ├── error_analysis.py                   # ДЗ3: типы ошибок, срезы
    ├── robustness.py                       # ДЗ3: проверки устойчивости
    └── utils.py                            # конфиги, сиды, метрики
```