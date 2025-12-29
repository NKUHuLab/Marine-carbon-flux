import pandas as pd
import xgboost as xgb
import joblib
df = pd.read_parquet("Filtered_data_main.parquet")
model = joblib.load("XGB_model.pkl")
X = df.iloc[:, 9:]
d_full_data = xgb.DMatrix(X)
booster = model.get_booster()
shap_values_xgb = booster.predict(d_full_data, pred_contribs=True)
shap_columns = [f"SHAP_{col}" for col in X.columns]
shap_df = pd.DataFrame(shap_values_xgb[:, :-1], columns=shap_columns)
result = pd.concat([df, shap_df], axis=1)
result.to_parquet("SHAP_results.parquet")