# Models

`demo_ctr_pipeline.joblib` is a small **demo artifact** trained on generated data with the same feature schema. It is included so the Flask interface can be demonstrated immediately after cloning.

For the actual project results, run `notebooks/01_data_exploration.ipynb`, `02_feature_engineering.ipynb`, and `03_model_training.ipynb` on KASANDR. Notebook 03 writes `ctr_pipeline.joblib` and the Flask app will automatically use that real model when it exists.
