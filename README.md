Traffic Demand Prediction

This Repo contains the prediction file and source code necessary to reproduce the results submitted for the Traffic Demand Prediction competition.

Contents
- submission.csv — Final prediction file (Index, demand)
- source_files.zip — Archive containing APPROACH.txt and model source files
- APPROACH.txt — Detailed methodology and feature engineering
- ultra_advanced_model.py — Best model (4-model stacking)
- advanced_traffic_model.py — Backup model
- traffic_demand_model.py — Reference baseline model

Quick start
1. To regenerate predictions locally, install dependencies:
   pip install pandas numpy scikit-learn xgboost lightgbm catboost
2. Run the best model (may take several minutes):
   python ultra_advanced_model.py
3. Or use precomputed predictions: load `submission.csv` with pandas.

Notes
- The dataset is not included in this package. Obtain `train.csv` and `test.csv` from the competition and place them in a `dataset/` folder alongside these files if you want to re-train.
- Replace placeholder author/year in LICENSE as appropriate before publishing.

License
This repository is licensed under the MIT License — see LICENSE for details.
