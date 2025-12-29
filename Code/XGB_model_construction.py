import pandas as pd
import numpy as np
import joblib
import random
from xgboost import XGBRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
def rmse(obs, pre):
    return np.sqrt(mean_squared_error(obs, pre))
def calculate_cor(model, x_train, x_test, y_train, y_test, colnames):
    y_test_pred = pd.DataFrame(model.predict(x_test).reshape(-1, 1), index=y_test.index)
    y_train_pred = pd.DataFrame(model.predict(x_train).reshape(-1, 1), index=y_train.index)
    r_test = r2_score(y_test[colnames[0]], y_test_pred[0])
    r_train = r2_score(y_train[colnames[0]], y_train_pred[0])
    rmse_test = rmse(y_test[colnames[0]], y_test_pred[0])
    rmse_train = rmse(y_train[colnames[0]], y_train_pred[0])
    mae_test = mean_absolute_error(y_test[colnames[0]], y_test_pred[0])
    mae_train = mean_absolute_error(y_train[colnames[0]], y_train_pred[0])
    return r_train, r_test, rmse_train, rmse_test, mae_train, mae_test
model = XGBRegressor(n_estimators=500, random_state=0, max_depth=5, tree_method='hist', device='cuda')
frame = pd.read_parquet("Filtered_data_main.parquet")
random.seed(0)
val = random.sample(range(len(frame)), 5)
model_index = [i for i in frame.index if i not in val]
x_data = frame.loc[model_index].iloc[:, 9:].reset_index(drop=True)
y_data = frame.loc[model_index].iloc[:, 8:9].reset_index(drop=True)
colnames = y_data.columns.tolist()
corlist_train, corlist_test, rmsel_train, rmsel_test, mae_list_train, mae_list_test = [], [], [], [], [], []
for train_index, test_index in ShuffleSplit(n_splits=5, test_size=0.3, random_state=0).split(x_data, y_data):
    x_train, y_train = x_data.iloc[train_index], y_data.iloc[train_index]
    x_test, y_test = x_data.iloc[test_index], y_data.iloc[test_index]
    model.fit(x_train, y_train.values.ravel())
    r_train, r_test, rmse_train, rmse_test, mae_train, mae_test = calculate_cor(model, x_train, x_test, y_train, y_test, colnames)
    corlist_train.append(r_train)
    corlist_test.append(r_test)
    rmsel_train.append(rmse_train)
    rmsel_test.append(rmse_test)
    mae_list_train.append(mae_train)
    mae_list_test.append(mae_test)
results_df = pd.DataFrame([{
    'n_estimators': 500,
    'max_depth': 5,
    'mean_train_r2': np.mean(corlist_train),
    'mean_test_r2': np.mean(corlist_test),
    'rmse_train': np.mean(rmsel_train),
    'rmse_test': np.mean(rmsel_test),
    'mean_mae_train': np.mean(mae_list_train),
    'mean_mae_test': np.mean(mae_list_test)
}])
results_df.to_excel("XGB_model.xlsx", index=False)
joblib.dump(model, "XGB_model.pkl")