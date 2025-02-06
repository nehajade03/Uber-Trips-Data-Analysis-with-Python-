#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


# Importing libraries
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import zscore


# # Load Dataset

# In[2]:


# Load the Uber trips Dataset
Uber_df = pd.read_csv(r"C:\Users\jades\Downloads\uber trips data analysis dataset.csv")
print(Uber_df.head(10))
print(Uber_df.shape)


# In[3]:


# Data Overview
print(Uber_df.shape)
print(Uber_df.info())
print(Uber_df.columns)


# In[4]:


# Summary Statistics
print(Uber_df.describe())


# # Missing value checking 

# In[5]:


# Data Cleaning: Check for Missing Values and Duplicates
print(Uber_df.duplicated().sum())
# Drop Duplicates
Uber_df.drop_duplicates(inplace=True)


# In[6]:


# Check for Missing Values
print(Uber_df.isnull().sum())


# In[7]:


# Automatically detect numerical columns
numerical_columns = Uber_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Numerical Columns:", numerical_columns)


# # Outliers Detection

# In[ ]:


# Check for outliers using Z-score
Uber_df.drop_duplicates(inplace=True)
# Handle missing values (example: dropping rows with missing values)
Uber_df.dropna(inplace=True)


# In[9]:


# Outlier detection using Z-score
numerical_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
z_scores = Uber_df[numerical_columns].apply(zscore)
outliers = (z_scores.abs() > 3).any(axis=1)
Uber_df_cleaned = Uber_df[~outliers]
print(f"Rows after removing outliers: {Uber_df_cleaned.shape[0]}")


# In[10]:


# Feature Engineering: Convert 'pickup_datetime' to datetime format
Uber_df_cleaned['pickup_datetime'] = pd.to_datetime(Uber_df_cleaned['pickup_datetime'], errors='coerce')
print(Uber_df_cleaned['pickup_datetime'].dtype)


# # Feature Engineering

# In[11]:


# Feature Engineering: Extract additional features from 'pickup_datetime'
Uber_df_cleaned['Date/Time'] = pd.to_datetime(Uber_df_cleaned['pickup_datetime'])
Uber_df_cleaned['hour'] = Uber_df_cleaned['pickup_datetime'].dt.hour
Uber_df_cleaned['day_of_week'] = Uber_df_cleaned['pickup_datetime'].dt.day_name()
Uber_df_cleaned['month'] = Uber_df_cleaned['pickup_datetime'].dt.month_name()


# # Exploratory Data Analysis (EDA)

# In[12]:


# Visualization: Trips by Hour of the Day (Bar Plot)
plt.figure(figsize=(8, 4))
sns.countplot(data=Uber_df_cleaned, x='hour')
plt.title('Number of Trips by Hour of the Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Trips')
plt.xticks(rotation=45, fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[13]:


# Visualization: Trip over Time (Fare Distribution)
sns.histplot(Uber_df_cleaned['fare_amount'], kde=True)
plt.title('Fare Amount Distribution')
plt.show()


# In[14]:


# Visualization: Trips by Day of the Week (Bar Plot)
trips_by_day = Uber_df_cleaned['day_of_week'].value_counts()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
trips_by_day = trips_by_day.reindex(days_order, fill_value=0)

plt.figure(figsize=(10, 6))
plt.bar(trips_by_day.index, trips_by_day.values, color='lightblue')
plt.title('Number of Uber Trips by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Trips')
plt.grid(True)
plt.show()


# In[15]:


# Visualization: Percent of Trips by Day of the Week (Pie Chart)
plt.figure(figsize=(8, 6))
plt.pie(trips_by_day.values, labels=trips_by_day.index, autopct='%1.1f%%', colors=sns.color_palette("Set3", n_colors=7), startangle=60)
plt.title('Distribution of Uber Trips by Day of the Week')
plt.show()


# In[16]:


# Visualization: Fare Amount vs Passenger Count (Scatter Plot)
plt.figure(figsize=(10,6))
sns.scatterplot(x='passenger_count', y='fare_amount', data=Uber_df_cleaned)
plt.title('Fare Amount vs. Passenger Count')
plt.xlabel('Passenger Count')
plt.ylabel('Fare Amount')
plt.show()


# In[17]:


# Visualization: Fare Amount by Hour of the Day (Box Plot)
plt.figure(figsize=(10,6))
sns.boxplot(x='hour', y='fare_amount', data=Uber_df_cleaned)
plt.title('Fare Amount by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Fare Amount')
plt.show()


# # Machine Learning Model -  Random Forest Regressor

# In[18]:


# Data Splitting: Define features (X) and target variable (y)
X = Uber_df_cleaned[['hour', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]
y = Uber_df_cleaned['fare_amount']


# In[19]:


# Split the data into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


# In[20]:


# Model Training: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[21]:


# Predict on the test set
y_pred = rf_model.predict(X_test)


# In[22]:


# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")


# In[23]:


# Model Evaluation: Feature Importance Visualization
feature_importance = rf_model.feature_importances_
features = X.columns
feature_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title('Feature Importance')
plt.show()


# In[24]:


# Visualization: Predictions vs Actual Fare Amount
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.title('Predicted vs Actual Fare Amount')
plt.xlabel('Actual Fare Amount')
plt.ylabel('Predicted Fare Amount')
plt.show()


# # Save the ML Model

# In[25]:


# Save the trained Random Forest model
import joblib
joblib.dump(rf_model, "uber_fare_model.pkl")

# Load the saved model and make predictions
rf_model = joblib.load("uber_fare_model.pkl")


# In[26]:


# Prediction Function
def predict_fare(hour, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude):
    # Define feature order
    feature_order = ['hour', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    
    # Create DataFrame in correct order
    sample_trip = pd.DataFrame([[hour, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude]],
                               columns=feature_order)

    # Predict fare
    predicted_fare = rf_model.predict(sample_trip)
    
    return round(predicted_fare[0], 2)


# # Prediction on sample data 

# In[32]:


# Example usage of the prediction function
predicted_price = predict_fare(13, 5, -73.9855, 40.7580, -74.0000, 40.7128)
print(f"Predicted Fare Amount: ${predicted_price}")


# In[ ]:




