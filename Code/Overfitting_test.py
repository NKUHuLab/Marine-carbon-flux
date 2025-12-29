import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
data = pd.read_parquet("Filtered_data_main.parquet")
results = []
ratio_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
n_i = 10
n_k = 10
total_iterations = len(ratio_values) * n_i * n_k
with tqdm(total=total_iterations) as pbar:
    for ratio in ratio_values:
        for i in range(n_i):
            data_permuted = data.copy()
            n_samples = int(ratio * len(data_permuted))
            indices = np.random.choice(len(data_permuted), n_samples, replace=False)
            y_min = data_permuted.iloc[:, 8].min()
            y_max = data_permuted.iloc[:, 8].max()
            data_permuted.iloc[indices, 8] = np.random.uniform(y_min, y_max, n_samples)
            original_permuted_cor = pearsonr(data_permuted.iloc[:, 8], data.iloc[:, 8])[0]
            fold_num = len(data_permuted) // 10
            disorder = np.random.permutation(len(data_permuted))
            for k in range(n_k):
                start_idx = fold_num * k
                end_idx = fold_num * (k + 1)
                test_indices = disorder[start_idx:end_idx]
                train_indices = np.delete(disorder, range(start_idx, end_idx))
                train_data = data_permuted.iloc[train_indices]
                test_data = data_permuted.iloc[test_indices]
                X_train = train_data.iloc[:, 9:].values
                y_train = train_data.iloc[:, 8].values
                X_test = test_data.iloc[:, 9:].values
                y_test = test_data.iloc[:, 8].values
                xgb_model_cv = XGBRegressor(
                    n_estimators=500,
                    max_depth=5,
                    objective='reg:squarederror',
                    tree_method='hist',
                    device='cuda',
                    random_state=42
                )
                xgb_model_cv.fit(X_train, y_train)
                pred = xgb_model_cv.predict(X_test)
                rss = np.sum((y_test - pred) ** 2)
                tss = np.sum((y_test - np.mean(y_test)) ** 2)
                q2 = 1 - rss / tss
                results.append({
                    'Q2': q2,
                    'Correlation': original_permuted_cor,
                    'Ratio': ratio
                })
                pbar.update(1)
results_df = pd.DataFrame(results)
z = np.polyfit(results_df['Correlation'], results_df['Q2'], 1)
intercept = z[1]
print(f"Intercept: {intercept}")