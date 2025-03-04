# MLflow Projects: Global Superstore & House Pricing

## Overview
This repository contains Jupyter Notebooks for training and deploying machine learning models using MLflow for two datasets:

1. **Global Superstore MLflow Project** - Analyzes and deploys a model based on the Global Superstore dataset.
2. **House Pricing MLflow Project** - Analyzes and deploys a house pricing prediction model.

---

# Global Superstore MLflow Project

## Notebook Summaries

### 1. Global_super_store_MLflow.ipynb
- **Purpose**: Prepares and trains a model while logging experiments with MLflow.
- **Key Features**:
  - Loads and explores the Global Superstore dataset.
  - Uses Pandas, Seaborn, and Matplotlib for data analysis.
  - Implements MLflow to track experiments and model performance.

### 2. Global_superstore_deploy.ipynb
- **Purpose**: Deploys a trained ML model using MLflow.
- **Key Features**:
  - Configures environment variables for MLflow tracking.
  - Searches for existing MLflow runs.
  - Deploys the trained model using `mlflow models serve`.

---

# House Pricing MLflow Project

## Notebook Summaries

### 1. House_Pricing_MLflow.ipynb
- **Purpose**: Prepares and trains a model while logging experiments with MLflow.
- **Key Features**:
  - Loads and explores the house pricing dataset.
  - Uses Pandas, Seaborn, and Matplotlib for data analysis.
  - Implements MLflow to track experiments and model performance.

### 2. HousePricing@Deploy.ipynb
- **Purpose**: Deploys a trained ML model using MLflow.
- **Key Features**:
  - Configures environment variables for MLflow tracking.
  - Searches for existing MLflow runs.
  - Deploys the trained model using `mlflow models serve`.

---

## Prerequisites
To run these notebooks, ensure you have the following installed:
- Python 3.x
- Jupyter Notebook
- MLflow
- Pandas, NumPy, Matplotlib, and Seaborn

---

## Usage
1. Run `Global_super_store_MLflow.ipynb` to train and track the Global Superstore model.
2. Run `Global_superstore_deploy.ipynb` to deploy the trained Global Superstore model.
3. Run `House_Pricing_MLflow.ipynb` to train and track the house pricing model.
4. Run `HousePricing@Deploy.ipynb` to deploy the trained house pricing model.

---

## Notes
- Update the dataset paths in respective training notebooks before running.
- Ensure MLflow is configured correctly before deployment.

For further details, refer to the respective notebooks.

