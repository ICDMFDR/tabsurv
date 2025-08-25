import os
import time
import pandas as pd
import numpy as np
import torch
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

from datasets import load_datafile, load_datafile_gene, load_tab_survival_dataset
from utils import manual_c_index_expected_time
from tabpfn import TabPFNRegressor

CLIN_DATASETS = []

GENE_DATASETS = ['TCGA500', "GEO",   'GSE6532',"GSE19783",
                 'HEL', 'unt', 'nki', "transbig",
                 'UK', 'mainz', 'upp', 'METABRIC']


All_DATASETS = CLIN_DATASETS

for name in GENE_DATASETS:
    All_DATASETS.append(name)
print(All_DATASETS)


TIMESTRING = time.strftime("%Y%m%d%H%M")
RESULTS_DIR = f"./output/TabSurv_results_{TIMESTRING}"
os.makedirs(RESULTS_DIR, exist_ok=True)
PREDICTION_DIR = f"./output/prediction"

model_name = "TabSurv"
test_size = 0.5
random_state = 42

np.random.seed(42)
_ = torch.manual_seed(42)

all_results = []
ci_list = []
hr_list = []

for dataset_name in All_DATASETS:
    print(f"Processing dataset: {dataset_name}")

    # Load dataset
    if dataset_name in ["metabric", "gbsg", "support", "flchain", "nwtco"]:
        df, cols_standardize, cols_leave, duration_col, event_col = load_datafile(dataset_name)
    else:
        df, cols_standardize, cols_leave, duration_col, event_col = load_datafile_gene(dataset_name)

    X_train, y_train, X_test, y_test, y_test_event = load_tab_survival_dataset(df, duration_col, event_col, test_size, random_state)

    # Train TabSurv model
    model = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu",
                           ignore_pretraining_limits=True)
    # model = TabPFNRegressor(ignore_pretraining_limits=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Prepare results DataFrame
    df_results = pd.DataFrame(X_test.copy())
    df_results["duration"] = y_test
    df_results["event"] = y_test_event
    df_results["predicted"] = y_pred
    
    df_results.to_csv(os.path.join(PREDICTION_DIR, f"{dataset_name}_{model_name}_predict.csv"), index=False)


    # === Compute Metrics ===
    # 1. C-index
    c_index_manual = manual_c_index_expected_time(
        df_results, time_col="duration", event_col="event", prediction_col="predicted"
    )

    # Since TabSurv predicts survival time (higher = lower risk),
    # use negative values as risk scores
    df_results["risk_score"] = -df_results["predicted"]
    print(f"C-index: {c_index_manual:.4f}")
    # Store per-dataset results
    ci_list.append(c_index_manual)
    all_results.append({
        "dataset": dataset_name,
        "C-index": c_index_manual,
    })

# === Save results per dataset ===
results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{RESULTS_DIR}/TabSurv_c_index_metrics.csv", index=False)

# === Compute Stability Scores across datasets ===
stability_ci = np.mean(ci_list) - np.std(ci_list)

stability_df = pd.DataFrame([{
    "model": model_name,
    "mean_C-index": np.mean(ci_list),
    "std_C-index": np.std(ci_list),
    "Stability_CI": stability_ci,

}])

stability_df.to_csv(f"{RESULTS_DIR}/TabSurv_stability.csv", index=False)

print("\n=== Final Stability Scores ===")
print(stability_df)
