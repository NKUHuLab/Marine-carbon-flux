import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import matplotlib
import matplotlib.colors as colors
import warnings
warnings.filterwarnings("ignore")
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
model_path = "XGB_model.pkl"
file_path = "Filtered_data_main.parquet"
xgb = joblib.load(model_path)
data = pd.read_parquet(file_path)
data = data[data['marine_region'] == "South Pacific Ocean"]
filtered_data = data.iloc[:, 8:]
X = filtered_data.drop(columns=['FCO2'])
y = filtered_data['FCO2']
feature_pair = ('Rx1day', 'TR')
x_var_name = feature_pair[0]
y_var_name = feature_pair[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
pdp_results = partial_dependence(
    xgb,
    X_train,
    features=feature_pair,
    grid_resolution=100
)
XX, YY = np.meshgrid(pdp_results['values'][0], pdp_results['values'][1])
Z = pdp_results['average'][0].T
levels = [-72, -46, -32, -11, 0, 21, 34, 72, 93]
cmap_colors = ["#4E68D7", "#779AF6", "#A4C2FE", "#CCD8F0", "#EDD3C6", "#F7B094", "#EB7D64", "#CB3E37"]
cmap = colors.ListedColormap(cmap_colors)
cmap.set_under("#3B4CC0")
cmap.set_over("#B50325")
norm = colors.BoundaryNorm(levels, cmap.N)
fig, ax = plt.subplots(figsize=(6, 5))
contourf = ax.contourf(XX, YY, Z, levels=levels, cmap=cmap, norm=norm, extend='both')
ax.set_xlabel(f"{x_var_name} (mm)", fontsize=22)
ax.set_ylabel(f"{y_var_name} (day)", fontsize=22)
ax.tick_params(axis='both', labelsize=20, direction='out', length=5, width=1, colors='black')
cbar = fig.colorbar(contourf, extend='both')
cbar.set_ticks(levels)
cbar.set_label('FCO2 (mmol·m-2·month-1)', fontsize=22)
cbar.ax.tick_params(labelsize=20)
fig.savefig("SPO_Rx1day_TR_PDP.pdf", dpi=600, bbox_inches='tight')
plt.close(fig)