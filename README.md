# UK-House-Price-Predictive-Model-and-Brexit-Analysis
This project aims to analyze UK house prices and build a predictive model to estimate house prices using machine learning algorithms. Additionally, the impact of Brexit on house prices is explored through data analysis and visualization.

# Dataset
The dataset used in this project is "UK_Housing_Data.csv". It contains information about UK property sales, including various features such as price, property type, location, and transaction date. This dataset can be found on Kaggle.

# Data Preprocessing
Corrected data types and values in certain columns.
Removed irrelevant columns and rows with missing values.
Encoded categorical variables using Target Encoder.
# Exploratory Data Analysis (EDA)
Investigated the relationships between house prices and other features.
Analyzed the impact of Brexit on house prices using visualizations.
# Building Predictive Models
Three different regression models were trained and evaluated to estimate house prices:

Linear Regression Model

KNeighbors Regressor Model

SGD Regressor Model
# Evaluation Metrics
Root Mean Squared Error (RMSE)

R-squared (R2) Score
# Files
UK_Housing_Data.csv: Dataset used in the analysis.

predictive_model_and_analysis.py: Python script for building, evaluating predictive models, and analyzing the impact of Brexit.
# Libraries Used
pandas

numpy

matplotlib

scikit-learn

category_encoders
# Usage
Install the required libraries: Copy code 'pip install -r requirements.txt'

Run the python file 'predictive_model_and_analysis.py' to reproduce the analysis and build predictive models.
# Acknowledgment
The dataset used in this project was obtained from Kaggle.
