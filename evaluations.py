import os
import time
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# Define  paths and models
TIMESTRING = time.strftime("%Y%m%d%H%M")
PREDICTION_DIR = f"./output/prediction"
OUTPUT_DIR = f"./output/evaluation_results_{TIMESTRING}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = ["TabSurv", "LogisticHazard", "PMF", "DeepHitSingle", "PCHazard", "MTLR", "DeepSurv", "RSF"]

ALL_DATASETS = ['TCGA500', "GEO",   'GSE6532',"GSE19783",
                'HEL', 'unt', 'nki', "transbig",
                'UK', 'mainz', 'upp', 'METABRIC']

all_metrics = []


# ===== Metric Functions =====
def manual_c_index(durations, events, risk_scores):
    concordant, comparable = 0, 0
    n = len(durations)
    for i in range(n):
        for j in range(n):
            if durations[j] > durations[i] and events[i] == 1:
                comparable += 1
                if risk_scores[j] < risk_scores[i]:
                    concordant += 1
                elif risk_scores[j] == risk_scores[i]:
                    concordant += 0.5
    return concordant / comparable if comparable > 0 else np.nan


for dataset in ALL_DATASETS:
    for model_name in MODELS:
        file_path = os.path.join(PREDICTION_DIR, f"{dataset}_{model_name}_predict.csv")
        if not os.path.exists(file_path):
            print(f"❌ The dataset  {dataset} is not allowed to upload publicly due to the agreement.")
            continue

        df = pd.read_csv(file_path)

        # Expected columns: duration, event, predicted
        # Adjust column names if needed
        if "duration" not in df.columns:
            if "real_survival" in df.columns:
                df.rename(columns={"real_survival": "duration"}, inplace=True)
        if "event" not in df.columns and "event_observed" in df.columns:
            df.rename(columns={"event_observed": "event"}, inplace=True)

        # If TabSurv → convert predicted survival time to risk score
        if model_name == "TabSurv":
            df["risk_score"] = -df["predicted_survival"] if "predicted_survival" in df.columns else -df["predicted"]
        else:
            df["risk_score"] = df["predicted_risk_scores"] if "predicted_risk_scores" in df.columns else df["predicted"]

        # Fit CoxPH model
        cph = CoxPHFitter()
        cox_df = df[["duration", "event", "risk_score"]].copy()
        try:
            cph.fit(cox_df, duration_col="duration", event_col="event")
            summary = cph.summary.loc["risk_score"]

            # C-index (using risk_score)
            c_index = concordance_index(df["duration"], -df["risk_score"], df["event"])

            print(f"CI_{dataset}_{model_name}: {c_index}")

            all_metrics.append({
                "Dataset": dataset,
                "Model": model_name,
                "C-index": c_index,

            })
        except Exception as e:
            print(f"⚠ Error fitting CoxPH for {dataset} - {model_name}: {e}")
            continue


        # # Ensure column names
        # if "duration" not in df.columns and "real_survival" in df.columns:
        #     df.rename(columns={"real_survival": "duration"}, inplace=True)
        # if "event" not in df.columns and "event_observed" in df.columns:
        #     df.rename(columns={"event_observed": "event"}, inplace=True)
        #
        # # Determine risk score
        # if model_name == "TabSurv":
        #     df["risk_score"] = -df.get("predicted_survival", df.get("predicted"))
        # else:
        #     df["risk_score"] = df.get("predicted_risk_scores", df.get("predicted"))
        #
        # durations = df["duration"].values
        # events = df["event"].values.astype(int)
        # risk_scores = df["risk_score"].values
        #
        # c_index = manual_c_index(durations, events, risk_scores)
        # print(f"CI_{dataset}_{model_name}: {c_index}")
        #
        # all_metrics.append({
        #     "Dataset": dataset,
        #     "Model": model_name,
        #     "C-index": c_index,
        # })


# Convert to DataFrame
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "c_index_metrics.csv"), index=False)

pivot_df = metrics_df.pivot_table(index="Dataset", columns="Model", values="C-index", aggfunc="mean")
# Optional: reorder rows and columns
pivot_df = pivot_df.reindex(ALL_DATASETS)
pivot_df = pivot_df[ [c for c in MODELS if c in pivot_df.columns] ]


# Reset index so 'model' is a column
pivot_df.reset_index(inplace=True)

# Save to CSV
pivot_df.to_csv(os.path.join(OUTPUT_DIR, "all_model_c_index_pivot.csv"), index=False, float_format="%.4f")


# Print result
print("\nFinal Results Table (C-index):")
print(pivot_df)

# Generate Summary Table (mean per model)
summary_df = metrics_df.groupby("Model").agg(
    mean_C_index=("C-index", "mean"),
    std_C_index=("C-index", "std"),
).reset_index()

summary_df["Stability_CI"] = summary_df["mean_C_index"] - summary_df["std_C_index"]

# Ensure custom order for models
summary_df["Model"] = pd.Categorical(summary_df["Model"], categories=MODELS, ordered=True)
summary_df = summary_df.sort_values("Model")

# Save CSV with "Model" as first column
output_path = os.path.join(OUTPUT_DIR, "stability_score.csv")
summary_df.to_csv(output_path, index=False, float_format="%.4f")

print(f"✅ Summary table saved to {output_path}")
print(summary_df)

