import os, random, warnings, re
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
import joblib
from multiprocessing import Pool
from Causal_analysis_function import single_estimate_worker
from sklearn.preprocessing import QuantileTransformer
warnings.filterwarnings("ignore")
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
model_pkl_path = r"XGB_model.pkl"
data_files = "Filtered_data_main.parquet"
target_region = 'South Pacific Ocean'
N_REPEATS = 10
N_BOOT = 1000
CONF_LVL = 0.95
num_processes = 30
def process_single_path(treatment, outcome, region, model_pkl_path, dataset, causal_graph):
    import gc
    region_data = dataset[dataset['marine_region'] == region].copy()
    tasks = []
    for i in range(N_REPEATS):
        tasks.append({
            'seed_offset': i,
            'model_pkl_path': model_pkl_path,
            'treatment': treatment,
            'outcome': outcome,
            'dataset': region_data,
            'causal_graph': causal_graph,
            'SEED': SEED
        })
    try:
        with Pool(processes=num_processes) as pool:
            results_iterator = pool.imap(
                single_estimate_worker,
                tasks,
                chunksize=max(1, len(tasks) // num_processes)
            )
            ate_list = list(results_iterator)
    finally:
        del tasks, region_data
        gc.collect()
    ate_list = [result for result in ate_list if result is not None]
    rng = np.random.default_rng(SEED)
    boot_samples = rng.choice(ate_list, size=(N_BOOT, len(ate_list)), replace=True)
    boot_means = np.mean(boot_samples, axis=1)
    lower = np.percentile(boot_means, (1 - CONF_LVL) / 2 * 100)
    upper = np.percentile(boot_means, (1 + CONF_LVL) / 2 * 100)
    mean = np.mean(ate_list)
    del boot_samples, boot_means
    gc.collect()
    return {
        'region': region,
        'treatment': treatment,
        'outcome': outcome,
        'lower': lower,
        'upper': upper,
        'mean': mean
    }
if __name__ == "__main__":
    paths = [
        ("SAL", "FCO2"),
        ("Rx1day", "FCO2"),
        ("Rx1day", "SAL"),
        ("PP", "FCO2"),
        ("DIC", "FCO2"),
        ("sfcWind", "FCO2"),
        ("TR", "FCO2"),
        ("Evs", "SAL"),
        ("MP", "PP"),
        ("DFe", "PP"),
        ("NO3", "PP"),
        ("TR", "PP"),
        ("sfcWind", "Evs"),
        ("PP", "DIC"),
        ("DOC", "DIC")
    ]
    causal_graph = f"""digraph {{{' '.join([f'"{a}" -> "{b}";' for a, b in paths])}}}"""
    dataset = pd.read_parquet(data_files)
    dataset = dataset[dataset['marine_region'].isin(['South Pacific Ocean', 'South Atlantic Ocean'])]
    columns_to_transform = dataset.columns[8:]
    transformer1 = QuantileTransformer(output_distribution='uniform', random_state=42)
    dataset[columns_to_transform] = transformer1.fit_transform(dataset[columns_to_transform])
    transformer2 = QuantileTransformer(output_distribution='normal', random_state=42)
    dataset[columns_to_transform] = transformer2.fit_transform(dataset[columns_to_transform])
    region_summaries = []
    for treatment, outcome in tqdm(paths):
        result = process_single_path(
            treatment, outcome, target_region,
            model_pkl_path, dataset, causal_graph
        )
        if result:
            region_summaries.append({
                'Path': f"{treatment} -> {outcome}",
                'Lower Bound': result['lower'],
                'Upper Bound': result['upper'],
                'Mean': result['mean']
            })
    summary_df = pd.DataFrame(region_summaries)
    summary_df.to_excel("SPO_causal_results.xlsx", index=False)