import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
import joblib
import matplotlib
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
parquet_path = "Filtered_data_main.parquet"
model_path = "XGB_model.pkl"
region_name = 'South Pacific Ocean'
yticks = [-30, 0, 30, 60, 90]
precipitation_indicator = "Rx1day"
pdp_target = 'FCO2'
data = pd.read_parquet(parquet_path)
xgb_model = joblib.load(model_path)
all_columns = data.columns.tolist()
target_column = all_columns[8]
feature_columns = all_columns[9:]
region_data = data[data['marine_region'] == region_name].copy()
columns_to_use = [target_column] + feature_columns
filtered_data = region_data[columns_to_use].dropna()
X = filtered_data[feature_columns]
y = filtered_data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pdp_results = partial_dependence(
    estimator=xgb_model,
    X=X_train,
    features=[precipitation_indicator],
    grid_resolution=100
)
x_values_original = pdp_results['values'][0]
pd_values_original = pdp_results['average'][0]
feature_idx = feature_columns.index(precipitation_indicator)
X_temp = X_train.copy()
X_temp.iloc[:, feature_idx] = 0.0
predictions_at_zero = xgb_model.predict(X_temp)
pd_value_at_zero = predictions_at_zero.mean()
x_values = np.concatenate([[0.0], x_values_original])
pd_values = np.concatenate([[pd_value_at_zero], pd_values_original])
sort_idx = np.argsort(x_values)
x_values = x_values[sort_idx]
pd_values = pd_values[sort_idx]
n_splines = min(20, len(x_values) - 3)
gam_model = LinearGAM(s(0, n_splines=n_splines)).fit(x_values.reshape(-1, 1), pd_values)
r_squared = gam_model.statistics_['pseudo_r2']['explained_deviance']
p_value = gam_model.statistics_['p_values'][0]
if p_value < 0.01:
    p_text = "< 0.01"
elif p_value < 0.05:
    p_text = "< 0.05"
else:
    p_text = f"= {p_value:.2f}"
fig, ax = plt.subplots(figsize=(6, 5))
XX = gam_model.generate_X_grid(term=0, n=200)
pred_smooth = gam_model.predict(XX)
intervals = gam_model.confidence_intervals(XX, width=0.95)
ax.plot(XX, pred_smooth, color='#B11647', linewidth=2.5, label='GAM Fit')
ax.fill_between(XX.flatten(), intervals[:, 0], intervals[:, 1],
                color="#ED6246", alpha=0.25)
ax.grid(True, linestyle='--', dashes=(10, 10), linewidth=0.5, alpha=0.8, color='gray')
ax.set_axisbelow(True)
ax.set_xlim(0, None)
ax.set_yticks(yticks)
ax.set_xlabel('Rx1day (mm)', fontsize=22)
ax.set_ylabel('FCO2 (mmol·m-2·month-1)', fontsize=22)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
legend_text = f'R2 = {r_squared:.2f}, p {p_text}'
ax.legend([legend_text], loc='best', frameon=False,
          prop={'family': 'Times New Roman', 'size': 20})
plt.savefig("SPO_Rx1day_PDP.pdf", format='pdf', bbox_inches='tight')
plt.close(fig)