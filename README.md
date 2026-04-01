## Car price predictor ##


Used Car Price Prediction & Analysis (Quikr)
This project focuses on cleaning, analyzing, and building a machine learning model to predict the resale prices of used cars based on historical data scraped from Quikr. The dataset initially contained significant "noise" (dirty data), and a major portion of this project is dedicated to advanced data cleaning and preprocessing using Python.

# Project Overview
Used car prices depend on several factors such as the manufacturer, age of the vehicle, fuel type, and total kilometers driven. This project explores these relationships and implements a pipeline to clean raw data and prepare it for predictive modeling.

# Dataset Features
The raw dataset (carprice.csv) contains 892 rows and 6 columns:

name: The full model name of the car.

company: The automobile manufacturer (e.g., Hyundai, Maruti, Ford).

year: The manufacturing year.

Price: The selling price (Target Variable).

kms_driven: Total distance the car has travelled.

fuel_type: Type of fuel used (Petrol, Diesel, LPG).

# Data Cleaning Challenges & Solutions
The raw data was "dirty" and required several steps to become usable:

Year: Removed non-numeric values and converted from object to int.

Price: Filtered out rows with "Ask For Price," removed commas, and converted to int.

Kilometers Driven: Stripped the " kms" suffix, removed commas, and handled non-numeric entries.

Fuel Type: Removed rows with missing (NaN) values.

Name Consistency: Standardized car names by keeping only the first three words to reduce model complexity and grouping.

# Tech Stack
Language: Python

Libraries:

Pandas & NumPy for data manipulation.

Matplotlib & Seaborn for data visualization.

Scikit-Learn for building the machine learning pipeline (Linear Regression).

# Key Insights from Analysis
The data contains cars from 25 unique companies, with Maruti being the most frequent.

The average price of used cars in the cleaned dataset is approximately ₹4.11 Lakhs.

The dataset covers manufacturing years from 1995 to 2019.