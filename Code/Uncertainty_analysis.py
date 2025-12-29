import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
parquet_path = "Filtered_data_main.parquet"
noise_level = 0.01
test_size = 0.2
estimator_change = 25
num_simulations = 100
best_params_base = {
    'max_depth': 5,
    'n_estimators': 500,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42
}
label = "FCO2"
attributes = ["DFe", "DIC", "DOC", "Evs", "MP", "NO3", "PP", "SAL", "sfcWind", "CDD", "PRCPTOT", "R50mm", "R99p", "Rx1day", "SU", "TR"]
data = pd.read_parquet(parquet_path)
data_meta = data.iloc[:, :8].copy()
data_vars = data.iloc[:, 8:].copy()
def prepare_fixed_split(data, label, attributes, test_size=0.2):
    valid_mask = data[label].notna()
    data_clean = data[valid_mask].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        data_clean[attributes], data_clean[label],
        test_size=test_size, random_state=42
    )
    return x_train, y_train, x_test, y_test
def add_noise_to_data(data, noise_level, attributes):
    noisy_data = data.copy()
    for col in attributes:
        if col in noisy_data.columns:
            std = noisy_data[col].std()
            noise = np.random.normal(loc=0, scale=noise_level * std, size=len(noisy_data))
            noisy_data[col] += noise
    return noisy_data
def generate_random_params(param_ranges, base_params, seed):
    np.random.seed(seed)
    params = base_params.copy()
    for param, (lower, upper) in param_ranges.items():
        params[param] = int(np.random.uniform(lower, upper))
    params['random_state'] = seed
    return params
def train_and_predict_with_noise(x_train, y_train, params, predict_data,
                                 noise_level, attributes, scaler, seed):
    np.random.seed(seed)
    model = XGBRegressor(**params)
    x_train_scaled = scaler.transform(x_train[attributes])
    x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=attributes)
    model.fit(x_train_scaled_df, y_train)
    predict_data_copy = add_noise_to_data(predict_data, noise_level, attributes)
    predict_data_scaled = scaler.transform(predict_data_copy[attributes])
    predicted_label = model.predict(predict_data_scaled)
    return predicted_label
x_train_fixed, y_train_fixed, x_test_fixed, y_test_fixed = prepare_fixed_split(
    data_vars, label, attributes, test_size
)
base_scaler = StandardScaler()
base_scaler.fit(x_train_fixed[attributes])
param_ranges = {
    'max_depth': (best_params_base['max_depth']-1,
                 best_params_base['max_depth']+1),
    'n_estimators': (best_params_base['n_estimators'] - estimator_change,
                    best_params_base['n_estimators'] + estimator_change)
}
FCO2_simulations = []
predict_data = data_vars[attributes]
for i in tqdm(range(num_simulations)):
    params = generate_random_params(param_ranges, best_params_base, seed=i)
    x_train_noisy = add_noise_to_data(x_train_fixed, noise_level, attributes)
    predicted_FCO2 = train_and_predict_with_noise(
        x_train_noisy, y_train_fixed, params, predict_data,
        noise_level=noise_level, attributes=attributes,
        scaler=base_scaler, seed=i)
    FCO2_simulations.append(predicted_FCO2)
FCO2_simulations = np.array(FCO2_simulations)
FCO2_mean = np.mean(FCO2_simulations, axis=0)
FCO2_std = np.std(FCO2_simulations, axis=0)
FCO2_simulations_std_mean_ratio = np.where(np.abs(FCO2_mean) > 1e-6, FCO2_std / np.abs(FCO2_mean),0)
result_df = data_meta.copy()
result_df['FCO2_simulations_std_mean_ratio'] = FCO2_simulations_std_mean_ratio
target_col = result_df.columns[-1]
median = result_df[target_col].median()
mad = np.median(np.abs(result_df[target_col] - median))
modified_z_scores = 0.6745 * (result_df[target_col] - median) / mad
filtered_df = result_df[np.abs(modified_z_scores) < 3]
def calc_stats(data):
    return [data.mean() * 100, data.median() * 100, data.sem() * 100]
stats_data = []
stats_data.append(['Total'] + calc_stats(filtered_df[target_col]))
spo_data = filtered_df[filtered_df['marine_region'] == 'South Pacific Ocean'][target_col]
stats_data.append(['South Pacific Ocean'] + calc_stats(spo_data))
sao_data = filtered_df[filtered_df['marine_region'] == 'South Atlantic Ocean'][target_col]
stats_data.append(['South Atlantic Ocean'] + calc_stats(sao_data))
final_stats_df = pd.DataFrame(stats_data, columns=['Region', 'Mean', 'Median', 'SE'])
final_stats_df.to_excel("Uncertainty_results.xlsx", index=False)