import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from datetime import datetime
from sklearn.preprocessing import PowerTransformer
import warnings
warnings.filterwarnings('ignore')
def transform_yeo_johnson(base_data, columns_to_transform):
    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    result = base_data.copy()
    result[columns_to_transform] = transformer.fit_transform(result[columns_to_transform])
    return result
def parse_time_column(time_str):
    year = int(str(time_str)[:4])
    month = int(str(time_str)[4:6])
    return pd.Timestamp(year=year, month=month, day=1)
def prepare_time_series_data(df, columns_needed):
    df['datetime'] = df['Time'].apply(parse_time_column)
    df = df.dropna(subset=['datetime'])
    grouped = df.groupby('datetime')[columns_needed].mean()
    grouped = grouped.sort_index()
    return grouped
data_full = pd.read_parquet("Filtered_data_STA.parquet")
columns_to_transform = data_full.columns[-4:].tolist()
data_transformed = transform_yeo_johnson(data_full, columns_to_transform)
precip_var = "Rx1day"
ocean_vars = ["SAL", "ALK"]
target_var = "FCO2"
region_label = "South Pacific Ocean"
data_region = data_transformed[data_transformed['marine_region'] == region_label].copy()
for ocean_var in ocean_vars:
    columns_needed = [precip_var, ocean_var, target_var]
    time_series_df = prepare_time_series_data(data_region[['Time'] + columns_needed], columns_needed)
    df_clean = time_series_df.dropna()
    returns = np.log(df_clean / df_clean.shift(1)).dropna()
    returns_cleaned = returns[~returns.isin([np.inf, -np.inf]).any(axis=1)]
    model_var = VAR(endog=returns_cleaned)
    lags = range(1, min(10, len(returns_cleaned) // 3))
    best_lag = model_var.select_order(maxlags=max(lags)).aic
    results = model_var.fit(best_lag)
    irf = results.irf(10)
    fig_original = irf.plot_cum_effects(orth=True, figsize=(12, 12))
    keep_indices = [3, 7]
    chain_titles = [f"{precip_var} → {ocean_var}", f"{ocean_var} → {target_var}"]
    curve_data = pd.DataFrame()
    for i, (original_idx, title) in enumerate(zip(keep_indices, chain_titles)):
        original_ax = fig_original.axes[original_idx]
        x_data = original_ax.lines[0].get_xdata()
        y_center = original_ax.lines[0].get_ydata()
        y_lower = original_ax.lines[1].get_ydata()
        y_upper = original_ax.lines[2].get_ydata()
        curve_data[f'{title}_x'] = pd.Series(x_data)
        curve_data[f'{title}_y'] = pd.Series(y_center)
        curve_data[f'{title}_y_lower_ci'] = pd.Series(y_lower)
        curve_data[f'{title}_y_upper_ci'] = pd.Series(y_upper)
    plt.close(fig_original)
    plt.close('all')
    xlsx_filename = f"SPO_{ocean_var}_STA.xlsx"
    curve_data.to_excel(xlsx_filename, index=False)