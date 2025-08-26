import os
import time
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from datasets import load_datafile, load_datafile_gene, preprocess_dataset, get_target
from models import get_model, model_dict
from utils import get_labtrans, evaluate_model_sksurv
import torchtuples as tt
from sksurv.ensemble import RandomSurvivalForest

CLIN_DATASETS = []
GENE_DATASETS = ['TCGA500', "GEO",   'GSE6532',"GSE19783",
                 'HEL', 'unt', 'nki', "transbig",
                 'UK', 'mainz', 'upp', 'METABRIC']

MODELS = ["LogisticHazard", "PMF", "DeepHitSingle", "PCHazard", "MTLR", "DeepSurv", "RSF"]

TIMESTRING = time.strftime("%Y%m%d%H%M")
PREDICTION = f"./output/prediction_{TIMESTRING}/"
os.makedirs(PREDICTION, exist_ok=True)

np.random.seed(42)
_ = torch.manual_seed(42)

test_size = 0.5
random_state = 42

All_DATASETS = CLIN_DATASETS.copy()
for name in GENE_DATASETS:
    All_DATASETS.append(name)

all_results = []

for dataset_name in All_DATASETS:
    print(f"Processing dataset: {dataset_name}")

    file_path = f"input/{dataset_name}.csv"
    if not os.path.exists(file_path):
        print(f"❌ The dataset {dataset_name} is  available upon request.")
        continue
    df, cols_standardize, cols_leave, duration_col, event_col = load_datafile_gene(dataset_name)

    df_train, df_val, df_test, x_train, x_val, x_test, x_mapper = preprocess_dataset(
        df, cols_standardize, cols_leave, duration_col, event_col, test_size, random_state)

    for model_name in MODELS:
        print(f"  Training model: {model_name}")
        model_class = model_dict[model_name]
        labtrans = get_labtrans(model_class, 10) if model_name in ["LogisticHazard", "PMF", "DeepHitSingle", "PCHazard", "MTLR"] else None

        if labtrans is not None:
            y_train = labtrans.fit_transform(*get_target(df_train, duration_col, event_col))
            y_val = labtrans.transform(*get_target(df_val, duration_col, event_col))
            durations_test, events_test = get_target(df_test, duration_col, event_col)
            model = get_model(model_name, x_train.shape[1], labtrans.out_features, labtrans)
        else:
            durations_train, events_train = get_target(df_train, duration_col, event_col)
            durations_val, events_val = get_target(df_val, duration_col, event_col)
            durations_test, events_test = get_target(df_test, duration_col, event_col)
            y_train = (durations_train, events_train)
            y_val = (durations_val, events_val)
            model = get_model(model_name, x_train.shape[1])

        # Train model
        if model_name == "RSF":
            model = RandomSurvivalForest(
                n_estimators=200,
                min_samples_split=10,
                min_samples_leaf=15,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            )
            y_train_rsf = np.array(
                [(bool(e), t) for e, t in zip(df_train[event_col], df_train[duration_col])],
                dtype=[(event_col, 'bool'), (duration_col, 'float')]
            )
            model.fit(x_train, y_train_rsf)
        else:
            callbacks = [tt.cb.EarlyStopping()]
            model.fit(x_train, y_train, batch_size=256, epochs=100,
                      callbacks=callbacks, val_data=(x_val, y_val))

        # Evaluate model
        metrics = evaluate_model_sksurv(model, x_test, durations_test, events_test, model_name)
        df_results = metrics["df_results"]

        c_index = metrics["c_index"][0]
        print(f"C-index_{dataset_name}_{model_name}: {c_index}")

        df_results_rename = df_results.rename(columns={
            "real_survival": "duration",
            "event": "event",
            "predicted_risk_scores": "predicted"
        })
        df_results.to_csv(os.path.join(f"{PREDICTION}/{dataset_name}_{model_name}_predict.csv"), index=False)


        all_results.append({
            "dataset": dataset_name,
            "model": model_name,
            "C-index": metrics["c_index"][0],

        })


