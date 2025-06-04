The repository contains the data and codes accompanying the manuscript "Decreases in South Pacific and South Atlantic sea-air CO2 fluxes caused by extreme precipitation".

(1) System requirements
All machine learning analyses were performed in Python 3.10 on Windows 10.

The key Python libraries and their versions used in the analysis are as follows:
scikit-learn (1.4.2), xgboost (3.0.1), shap (0.44.1), dowhy (0.8), geatpy (2.7.0), statsmodels (0.13.5).

No non-standard hardware is required.

(2) Installation guide
All the abovementioned Python libraries can be installed using "pip install".

Typical installation time is under 10 minutes.

(3) Demo
Download the code and data, and save them all on your computer. Modify the folder paths in the code as needed, and then run the code.

For the code "XGB model construction.py", the expected output is a .xlsx file that records the performance of the models. The model can be saved as a .pkl file.

For the code "SHAP analysis.py",  the expected output is a .xlsx file containing SHAP values for all the input variables.

For the code "Univariate partial dependence analysis.py", the expected output is a .pdf file displaying the impact of  PRCPTOT on FCO2.

For the code "Bivariate partial dependence analysis.py", the expected output is a .pdf file displaying the impact of ALK and SAL on FCO2.

For the code "Causal analysis.py", the expected output is a .xlsx file containing the causal inference results for each causal path.

For the code "NSGA-II optimization algorithm.py", the expected output is a .xlsx file including the adjusted input to minimize FCO2 and the corresponding predicted FCO2.

For the code "Shock transmission analysis.py", the expected output is a .pdf file displaying the shock effects of PRCPTOT on ALK and the subsequent transmission-based influence of ALK on FCO2.

For the code "Uncertainty analysis.py", the expected output is a .xlsx file containing the percentage of model uncertainty.

The expected runtime of all the code above is more than 24 hours.

(4) Instructions for use
When executed following the steps above, the code will automatically process the data used in the manuscript and supplementary information.

The generated output will fully replicate the results presented in the manuscript.