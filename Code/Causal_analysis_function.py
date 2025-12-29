import numpy as np
import pandas as pd
import random
import re
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
import dowhy
import joblib
import warnings
warnings.filterwarnings("ignore")
def single_estimate_worker(task_dict):
    try:
        import gc
        seed_offset = task_dict['seed_offset']
        model_pkl_path = task_dict['model_pkl_path']
        treatment = task_dict['treatment']
        outcome = task_dict['outcome']
        dataset = task_dict['dataset']
        causal_graph = task_dict['causal_graph']
        SEED = task_dict['SEED']
        np.random.seed(SEED + seed_offset)
        random.seed(SEED + seed_offset)
        model_pkl = joblib.load(model_pkl_path)
        model = dowhy.CausalModel(
            data=dataset,
            graph=causal_graph.replace("/n", " "),
            treatment=treatment.strip('"'),
            outcome=outcome.strip('"')
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        cv_local = KFold(n_splits=5, shuffle=True, random_state=SEED + seed_offset)
        dml_est = model.estimate_effect(
            estimand,
            method_name="backdoor.econml.dml.DML",
            control_value=0,
            treatment_value=1,
            confidence_intervals=False,
            method_params={
                "init_params": {
                    "model_y": model_pkl,
                    "model_t": model_pkl,
                    "model_final": LassoCV(fit_intercept=False, random_state=SEED + seed_offset, cv=3),
                    "featurizer": PolynomialFeatures(degree=2, include_bias=True)
                },
                "fit_params": {},
                "cv": cv_local
            }
        )
        mean_val = float(re.findall(r"Mean value: ([-+eE0-9.]+)", str(dml_est))[0])
        del model_pkl, model, estimand, dml_est
        gc.collect()
        return mean_val
    except Exception as e:
        print(f"Error in seed_offset {task_dict.get('seed_offset', 'unknown')}: {str(e)}")
        import gc
        gc.collect()
        return None