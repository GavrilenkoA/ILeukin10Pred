import pandas as pd
from pycaret.classification import *
from pycaret.utils import check_metric

# 1) Данные
train = pd.read_csv('Il_10_AZUE_Transformed_Dataset.csv')

# 2) Setup без интерактива и лишних «тяжёлых» стадий
clf1 = setup(
    data=train,
    target='Class',
    train_size=0.80,
    fold=3,                    # быстрее, чем 5–10
    data_split_stratify=True,
    feature_selection=False,   # отключаем дорогой FS
    normalize=False,
    remove_multicollinearity=False,
    fix_imbalance=False,       # включайте только при необходимости (медленнее)
    session_id=123,
    silent=True,               # без input()
    verbose=False,
    log_experiment=False       # без MLflow и пр.
    # use_gpu=True,            # можно включить, если есть GPU (CatBoost/XGB)
)

# 3) Сравнение только нескольких кандидатов (без полного зоопарка)
candidates = ['lightgbm', 'catboost', 'et']  # ExtraTrees
best = compare_models(include=candidates, sort='AUC')

# 4) Обучение моделей по отдельности
lgbm = create_model('lightgbm')
et   = create_model('et')
cat  = create_model('catboost')

# 5) Лёгкий тюнинг LightGBM (ограничим количество итераций поиска)
lgbm_tuned = tune_model(lgbm, optimize='AUC', choose_better=True, n_iter=15)

# 6) Оценка на holdout
pred_holdout = predict_model(lgbm_tuned)

# 7) Сохраним графики на диск (в текущую папку)
plot_model(lgbm_tuned, plot='auc', save=True)
plot_model(lgbm_tuned, plot='confusion_matrix', save=True)
plot_model(lgbm_tuned, plot='feature', save=True)

# 8) Экспорт финальной модели (пикл + трансформы PyCaret)
final = finalize_model(lgbm_tuned)
save_model(final, 'il10_lgbm_final_pycaret23')


print("\n=== Holdout Metrics ===")
for metric in ['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa', 'MCC']:
    value = check_metric(pred_holdout['Class'], pred_holdout['Label'], metric)
    print(f"{metric:10s}: {value:.4f}")
