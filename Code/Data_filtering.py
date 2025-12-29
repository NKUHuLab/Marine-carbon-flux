import pandas as pd
file_path = "1990-2023_data.parquet"
df = pd.read_parquet(file_path)
df_dropna = df.dropna().reset_index(drop=True)
cols1 = [
    "Time", "Latitude", "Longitude", "Year", "Month", "Lat_index", "Lon_index", "marine_region",
    "FCO2", "DFe", "DIC", "DOC", "Evs", "MP", "NO3", "PP",
    "SAL", "sfcWind", "CDD", "PRCPTOT", "R50mm", "R99p", "Rx1day", "SU", "TR"
]
df_main = df_dropna[cols1].copy()
df_main.to_parquet("Filtered_data_main.parquet")
cols2 = [
    "Time", "Latitude", "Longitude", "Year", "Month", "Lat_index", "Lon_index",
    "marine_region", "FCO2", "ALK", "SAL", "Rx1day"
]
target_regions = ["South Pacific Ocean", "South Atlantic Ocean"]
df_sta = df_dropna.loc[df_dropna["marine_region"].isin(target_regions),cols2].copy()
df_sta.to_parquet("Filtered_data_STA.parquet")