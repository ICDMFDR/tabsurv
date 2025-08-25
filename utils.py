import numpy as np
import pandas as pd

from pycox.evaluation import EvalSurv

def get_labtrans(model_class, num_durations):
    return model_class.label_transform(num_durations)


import pandas as pd

def manual_c_index_expected_time(df, time_col='real_survival', event_col='event', prediction_col='predicted_survival'):
    """
    Manually compute C-index when the model predicts expected survival time
    (higher predicted value = better survival / lower risk).
    """
    df = df.reset_index(drop=True)  # Ensure row indices are 0, 1, ..., n-1

    n_concordant = 0
    n_discordant = 0
    n_tied = 0
    n_comparable = 0

    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue

            t_i, e_i, p_i = df.loc[i, time_col], df.loc[i, event_col], df.loc[i, prediction_col]
            t_j, e_j, p_j = df.loc[j, time_col], df.loc[j, event_col], df.loc[j, prediction_col]

            # Only compare if i had an event and lived less than j
            if t_i < t_j and e_i == 1:
                n_comparable += 1
                if p_i < p_j:       # Lower predicted survival → shorter real survival → concordant
                    n_concordant += 1
                elif p_i > p_j:
                    n_discordant += 1
                else:
                    n_tied += 1

    c_index_manual = (n_concordant + 0.5 * n_tied) / n_comparable if n_comparable > 0 else None
    # print(f"Manual C-index (manually): {c_index_manual:.4f}")
    print(f"Concordant: {n_concordant}, Discordant: {n_discordant}, Ties: {n_tied}, Comparable Pairs: {n_comparable}")
    return c_index_manual


def evaluate_model_pycox(model, x_test, durations_test, events_test, model_name):
    # Predict survival function
    if model_name == "PCHazard":
        surv = model.predict_surv_df(x_test)
    elif model_name == "DeepSurv":
        _ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(x_test)
    else:
        surv = model.interpolate(10).predict_surv_df(x_test)

    # Create evaluation object
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')

    # Compute predicted risk score: negative expected survival time (area under curve)
    time_grid = surv.index.values
    surv_np = surv.to_numpy().T  # shape: [n_samples, n_times]
    expected_survival_time = np.trapz(surv_np, time_grid, axis=1)
    risk_scores = -expected_survival_time  # higher risk = shorter expected survival

    # Convert x_test to DataFrame if needed
    if isinstance(x_test, np.ndarray):
        x_test = pd.DataFrame(x_test)

    # Create results DataFrame
    df_results = x_test.copy().reset_index(drop=True)
    df_results["real_survival"] = durations_test
    df_results["predicted_risk_scores"] = risk_scores
    df_results["event"] = events_test

    return {
        'c_index': ev.concordance_td('antolini'),
        'df_results': df_results
    }


##Using lifelines:
from lifelines.utils import concordance_index
def evaluate_model_lifelines(model, x_test, durations_test, events_test, model_name):


    if model_name == "RSF":
        risk_scores = model.predict(x_test)

    else:
        if model_name == "PCHazard":
            surv = model.predict_surv_df(x_test)
        elif model_name == "DeepSurv":
            _ = model.compute_baseline_hazards()
            surv = model.predict_surv_df(x_test)

        else:
            surv = model.interpolate(10).predict_surv_df(x_test)

        # Compute expected survival time (area under survival curve)
        time_grid = surv.index.values  # time points
        surv_np = surv.to_numpy().T  # shape: [n_samples, n_times]
        exp_surv_time = np.trapz(surv_np, time_grid, axis=1)
        risk_scores = -exp_surv_time  # higher risk = lower expected survival

    c_index = concordance_index(durations_test, risk_scores, event_observed=events_test)

    return {
        'c_index': c_index
    }



import numpy as np
from sksurv.metrics import concordance_index_censored

def evaluate_model_sksurv(model, x_test, durations_test, events_test, model_name):
    if model_name == "RSF":
        risk_scores = model.predict(x_test)

    else:
        if model_name == "PCHazard":
            surv = model.predict_surv_df(x_test)
        elif model_name == "DeepSurv":
            _ = model.compute_baseline_hazards()
            surv = model.predict_surv_df(x_test)
        elif model_name == "LogisticHazard":
            surv = model.predict_surv_df(x_test)

        else:
            # surv = model.interpolate(10).predict_surv_df(x_test, is_dataloader =True)
            surv = model.predict_surv_df(x_test)

        # Compute expected survival time (area under survival curve)
        time_grid = surv.index.values  # time points
        surv_np = surv.to_numpy().T  # shape: [n_samples, n_times]
        exp_surv_time = np.trapz(surv_np, time_grid, axis=1)
        risk_scores = -exp_surv_time  # higher risk = lower expected survival
        # risk_scores = exp_surv_time

    # Convert labels to structured array for sksurv
    from sksurv.util import Surv
    y_test_structured = Surv.from_arrays(events_test.astype(bool), durations_test)

    c_index_result = concordance_index_censored(
        y_test_structured["event"],
        y_test_structured["time"],
        risk_scores
    )
    # Convert x_test to DataFrame if it's a NumPy array
    if isinstance(x_test, np.ndarray):
        x_test = pd.DataFrame(x_test)
    # Create result DataFrame
    df_results = x_test.copy()
    df_results = df_results.reset_index(drop=True)
    df_results["real_survival"] = durations_test
    df_results["predicted_risk_scores"] = risk_scores
    df_results["event"] = events_test
    return {
        'c_index': c_index_result,
        'df_results': df_results

    }
