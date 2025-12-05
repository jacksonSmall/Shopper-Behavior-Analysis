# Shopper Behavior Analysis
**University of Central Florida | STA4724 - Big Data Analytics**

**Authors:** Jackson Small, Andres Machado, Thomas Tibbetts, Sarah Taha

## Project Links
* **Repository:** [GitHub - Shopper Behavior Analysis](https://github.com/jacksonSmall/Shopper-Behavior-Analysis)
* **Dataset:** [Online Shoppers Purchasing Intention Dataset (UCI)](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

## Project Overview
This project analyzes user session data to predict purchasing intention on an e-commerce platform. By leveraging supervised machine learning techniques, we aim to identify key behavioral indicators—such as page values and exit rates—that drive revenue generation.
The study compares multiple classification models to find the optimal balance between predictive accuracy (AUC) and business utility (Recall/Sensitivity).

## Methodology
Our analysis pipeline consisted of the following stages:
1.  **Exploratory Data Analysis (EDA):** Investigated distributions, correlations, and seasonality (e.g., Nov/Dec peaks).
2.  **Feature Engineering:**
    * **Clustering:** Applied K-Means (k=5) to create engineered feature `c5` to capture latent user groups.
    * **Transformations:** Log-transformed skewed features like `ProductRelated_Duration`.
3.  **Feature Selection:** Utilized ANOVA F-tests and Random Forest importance to identify `PageValues` and `ExitRates` as dominant predictors.
4.  **Modeling:** Evaluated four algorithms:
    * **K-Nearest Neighbors (KNN)**
    * **Logistic Regression**
    * **Random Forest (RF)**
    * **Support Vector Machine (SVM)**

## Project Structure
* `Final_EDA.ipynb`: Contains the initial data exploration, visualizations, and feature selection analysis.
* `model_tuning.ipynb`: The main modeling script containing training/testing for KNN, LogReg, RF, and SVM.
* `cluster_log_transform.py`: A custom Python module containing the `Cluster_Log` transformer class used in our pipelines.
* `ShopperBehaviorAnalysisReport.pdf`: The final academic report detailing findings.
