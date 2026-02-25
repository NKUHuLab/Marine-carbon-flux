This repository contains the data and code used for the analysis presented in the manuscript "Decreases in South Pacific and South Atlantic sea-air CO₂ fluxes caused by extreme precipitation" published in *Nature Communications* ([DOI: 10.1038/s41467-026-69847-6](https://doi.org/10.1038/s41467-026-69847-6)).

■ System Requirements

The analysis workflow was developed and tested on Windows 10 using Python 3.10. Major Python libraries used include scikit-learn (1.4.2), xgboost (3.0.1), dowhy (0.13), econml (0.15.0), statsmodels (0.14.5) and geatpy (2.7.0). Climate extreme indices (CEI) were calculated in R 4.3 using PCICt (0.5-4.4) and climdex.pcic (1.1-11) as the primary libraries.

■ Installation guide

Install Python requirements:

    pip install scikit-learn==1.4.2 xgboost==3.0.1 dowhy==0.13 econml==0.15.0 statsmodels==0.14.5 geatpy==2.7.0

Install R requirements:

    install.packages('http://cran.r-project.org/src/contrib/Archive/PCICt/PCICt_0.5-4.4.tar.gz', repos = NULL, type = "source")
    install.packages('http://cran.r-project.org/src/contrib/Archive/climdex.pcic/climdex.pcic_1.1-11.tar.gz', repos = NULL, type = "source")

Typical installation time is under 10 minutes.

■ Data Description

The analysis relies on the following datasets:

　➤"1990-2023_data.parquet": The global monthly dataset containing:

　　- Target: FCO2

　　- Features: physicochemical-biological properties (ALK, DFe, DIC, DOC, Evs, NO3, PP, SAL, sfcWind), microplastic concentration (MP) and climate extreme indices (CDD, PRCPTOT, R50mm, R99p, Rx1day, SU, TR).

　➤"Sample_data_CEI.xlsx": Contains daily maximum temperature, minimum temperature and precipitation sample data from 1981 to 2023. The 1981–2010 interval establishes the baseline to determine percentile-based thresholds for R99p.

　➤"2050_data.xlsx": Includes data for the South Pacific Ocean and South Atlantic Ocean in 2050 under SSP1-2.6 and SSP5-8.5 scenarios. This dataset is used to compare FCO₂ predictions between models that consider precipitation extreme indices and those that exclude them.

■ Code Description

The total expected runtime for all scripts exceeds 24 hours.

 ◆ 1. Data Preparation

　➤"CEI_calculation.R": Calculates climate extreme indices from daily temperature and precipitation data.

　　- Input: "Sample_data_CEI.xlsx"

　　- Output: "CEI_results.xlsx"

　➤"Data_filtering.py": Splits the data into a main global dataset and a dataset of the South Pacific Ocean and South Atlantic Ocean for the shock transmission analysis (STA).

　　- Input: "1990-2023_data.parquet"

　　- Output: "Filtered_data_main.parquet", "Filtered_data_STA.parquet"

◆ 2. Model Construction and Validation

　➤"XGB_model_construction.py": Constructs the XGB model to capture the relationships between FCO₂ and its driving factors.

　　- Input: "Filtered_data_main.parquet"

　　- Output: "XGB_model.pkl", "XGB_model.xlsx"

　➤"Uncertainty_analysis.py": Calculates uncertainty statistics for the global ocean, South Pacific Ocean and South Atlantic Ocean.

　　- Input: "Filtered_data_main.parquet"

　　- Output: "Uncertainty_results.xlsx"

　➤"Overfitting_test.py": Performs permutation tests to assess model overfitting by calculating the Q² intercept.

◆ 3. Model Interpretation

　➤"SHAP_analysis.py": Computes SHAP values to quantify feature importance.

　　- Input: "Filtered_data_main.parquet", "XGB_model.pkl"

　　- Output: "SHAP_results.parquet"

　➤"Univariate_partial_dependence_analysis.py": Visualizes the impact of Rx1day on FCO₂.

　　- Input: "Filtered_data_main.parquet", "XGB_model.pkl"

　　- Output: "SPO_Rx1day_PDP.pdf"

　➤"Bivariate_partial_dependence_analysis.py": Visualizes the combined effects of Rx1day and TR on FCO₂.

　　- Input: "Filtered_data_main.parquet", "XGB_model.pkl"

　　- Output: "SPO_Rx1day_TR_PDP.pdf"

◆ 4. Causal Analysis

　➤"Causal_analysis_main.py" and "Causal_analysis_function.py": Estimate the average treatment effect of specific causal paths (e.g., Rx1day → FCO₂).

　　- Input: "Filtered_data_main.parquet", "XGB_model.pkl"

　　- Output: "SPO_causal_results.xlsx"

◆ 5. Shock Transmission Analysis

　➤"Shock_transmission_analysis.py": Analyzes how a shock in Rx1day drives changes in salinity and alkalinity, which subsequently impact FCO₂.

　　- Input: "Filtered_data_STA.parquet"

　　- Output: "SPO_ALK_STA.xlsx", "SPO_SAL_STA.xlsx"

◆ 6. Optimization

　➤"NSGA-II_optimization.py": Utilizes the NSGA-II algorithm to maximize the marine carbon sink by optimizing precipitation extreme indices within a 20% range relative to 2023 values, while holding other variables constant.

　　- Input: "1990-2023_data.parquet", "XGB_model.pkl"

　　- Output: "Optimization_SPO_0.2.xlsx"