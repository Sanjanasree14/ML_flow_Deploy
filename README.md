# Overview
This repository contains a Machine Learning Workflow for Global Superstore sales data, covering data exploration, model training, and deployment using MLflow. The project follows an end-to-end machine learning pipeline, including data preprocessing, feature engineering, model training, evaluation, and model serving.

## Included Notebooks
Global Superstore MLflow (Global_super_store_MLflow.ipynb)

> Loads and explores the dataset.
> Performs data visualization and preprocessing for ML models.
## MLflow Model Deployment (Global_superstore_deploy.ipynb)

> Tracks machine learning experiments using MLflow.
> Deploys a trained Random Forest model using MLflow’s model serving.
## Dependencies
> Ensure you have the following libraries installed before running the notebooks:

bash
Copy
Edit
pip install numpy pandas matplotlib seaborn scikit-learn mlflow
Preprocessing Steps
Each project follows a structured data preprocessing pipeline, including:

Data Loading: Reads the dataset using Pandas.
Exploratory Data Analysis (EDA): Identifies trends and patterns in the data.
Handling Missing Values: Uses imputation strategies or removes null values.
Feature Engineering: Creates new features and encodes categorical variables.
Data Scaling: Normalizes numerical features for better model performance.
Model Training & Validation
Training and tracking models using MLflow.
Experimenting with Random Forest & other models.
Performing hyperparameter tuning to optimize accuracy.
Evaluating models using:
Classification Metrics: Accuracy, Precision, Recall, F1-score.
Regression Metrics: RMSE, MAE, R-squared.
Expected Results
Predictive Analytics: Forecast sales trends using ML models.
Feature Insights: Identify key factors affecting sales and profitability.
MLflow Experiment Tracking: Log and compare multiple model runs.
Model Deployment: Serve trained models using MLflow.
How to Use
Clone this repository and navigate to the respective notebook.
Run Global_super_store_MLflow.ipynb for data preprocessing and model training.
Log and track experiments using MLflow.
Deploy the trained model with Global_superstore_deploy.ipynb.
Run the MLflow model server with:
bash
Copy
Edit
mlflow models serve -p 8004 -m runs:/<your-run-id>/random-forest-model --no-conda
Replace <your-run-id> with the actual MLflow run ID.

References
MLflow Documentation
Pandas Documentation
Scikit-Learn Documentation
