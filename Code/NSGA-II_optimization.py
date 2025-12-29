import geatpy as ea
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
df_all = pd.read_parquet("1990-2023_data.parquet")
pkl_filename = "XGB_model.pkl"
all_columns = ['FCO2', 'DFe', 'DIC', 'DOC', 'Evs', 'MP', 'NO3', 'PP', 'SAL', 'sfcWind', 'CDD', 'PRCPTOT', 'R50mm', 'R99p', 'Rx1day', 'SU', 'TR']
output_col = 'FCO2'
input_cols = [col for col in all_columns if col != output_col]
model = joblib.load(pkl_filename)
df_all = df_all[df_all['Year'] == 2023]
cols_to_remove = ["Time", "Month"]
df_all = df_all.drop(columns=cols_to_remove, errors='ignore')
group_cols = ["Latitude", "Longitude", "Year"]
fco2_col_index = df_all.columns.get_loc("FCO2")
cols_to_average = df_all.columns[fco2_col_index:].tolist()
cols_before_fco2 = df_all.columns[:fco2_col_index].tolist()
other_cols = [col for col in cols_before_fco2 if col not in group_cols]
agg_dict = {}
for col in other_cols:
    agg_dict[col] = 'first'
for col in cols_to_average:
    agg_dict[col] = 'mean'
df = df_all.groupby(group_cols, as_index=False).agg(agg_dict)
df = df[cols_before_fco2 + cols_to_average]
required_cols = all_columns + ['marine_region']
df = df.dropna(subset=required_cols)
spo_data = df[df['marine_region'] == 'South Pacific Ocean'].copy()
adjust_vars = ["CDD", "PRCPTOT", "R50mm", "R99p", "Rx1day"]
adjustment_level = 0.20
data1 = spo_data[input_cols].reset_index(drop=True)
Compare = pd.DataFrame({output_col: spo_data[output_col].reset_index(drop=True)})
low_bound_original = data1.min(axis=0)
high_bound_original = data1.max(axis=0)
fixed_cols = [data1.iloc[:, i] for i in range(len(input_cols))]
result_dfY = pd.DataFrame(columns=[output_col])
result_dfX = pd.DataFrame(columns=input_cols)
for i in tqdm(range(data1.shape[0])):
    low_bound = low_bound_original.copy()
    high_bound = high_bound_original.copy()
    for idx, col in enumerate(input_cols):
        if col in adjust_vars:
            if col == "CDD":
                low_bound[col] = (1 - adjustment_level) * fixed_cols[idx][i]
                high_bound[col] = fixed_cols[idx][i]
            else:
                low_bound[col] = fixed_cols[idx][i]
                high_bound[col] = (1 + adjustment_level) * fixed_cols[idx][i]
        else:
            low_bound[col] = fixed_cols[idx][i]
            high_bound[col] = fixed_cols[idx][i]
    class MyProblem(ea.Problem):
        def __init__(self):
            name = 'NSGA-II'
            M = 1
            maxormins = [1]
            Dim = len(input_cols)
            varTypes = [0] * Dim
            lb = low_bound.values
            ub = high_bound.values
            lbin = [1] * Dim
            ubin = [1] * Dim
            super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        def evalVars(self, Vars):
            X = Vars
            f1 = model.predict(X)
            ObjV = f1[:, np.newaxis]
            CV = None
            return ObjV, CV
    problem = MyProblem()
    algorithm = ea.moea_NSGA2_templet(
        problem,
        ea.Population(Encoding='RI', NIND=100),
        MAXGEN=100,
        logTras=0
    )
    res = ea.optimize(algorithm, seed=1, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
    ObjV = res['ObjV']
    Vars = res['Vars']
    dfY = pd.DataFrame(ObjV)
    dfX = pd.DataFrame(Vars)
    max_percentage = float('-inf')
    max_index = None
    for a in range(len(dfY)):
        percentage = (Compare[output_col].iloc[i] - dfY.iloc[a, 0]) / Compare[output_col].iloc[i]
        if percentage > max_percentage:
            max_percentage = percentage
            max_index = a
    best_dfY = dfY.iloc[max_index]
    best_dfX = dfX.iloc[max_index]
    best_dfY = pd.DataFrame([best_dfY])
    best_dfX = pd.DataFrame([best_dfX])
    best_dfY.columns = [output_col]
    best_dfX.columns = input_cols
    FCO2_optimized = best_dfY[output_col].iloc[0]
    result_dfY = pd.concat([result_dfY, best_dfY], ignore_index=True)
    result_dfX = pd.concat([result_dfX, best_dfX], ignore_index=True)
result_combined = pd.concat([result_dfX, result_dfY], axis=1)
result_combined.to_excel("Optimization_SPO_0.2.xlsx", index=False)