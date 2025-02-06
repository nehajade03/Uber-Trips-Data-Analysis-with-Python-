# **Uber Fare Prediction**
A Machine Learning project that predicts Uber fare amounts using a Random Forest Regression model. The project covers data cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, deployment.
 ## **Features**

- Data Cleaning: Handling missing values, duplicates, and outliers.
- Exploratory Data Analysis (EDA): Visualizing trip patterns and fare distribution.
- Feature Engineering: Extracting hour, day_of_week, month from timestamps.
- Model Training: Random Forest Regressor for fare prediction.
- Model Evaluation: Metrics such as MAE, MSE, and R² Score.
- Model Deployment: Saving and reusing the trained model using joblib.

## **Exploratory Data Analysis (EDA)**
- Trips by Hour of the Day (Bar Chart)
- Trips by Day of the Week (Pie Chart)
- Scatter Plot: Passenger Count vs. Fare
- Box Plot: Fare Amount by Hour of the Day

## **Model Training & Evaluation**

    Algorithm Used: RandomForestRegressor
    
    Features Selected:
    
    hour, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude
    
    Performance Metrics:
    
        Mean Absolute Error (MAE)
        
        Mean Squared Error (MSE)
        
        R² Score
## **Model Deployment**

- Save the trained model using joblib:

 - Load the model and predict fares:
