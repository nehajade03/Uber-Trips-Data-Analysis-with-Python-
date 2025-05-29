# 🚖 Uber Fare Prediction & Analysis  

## 📌 Project Overview  
This project focuses on analyzing Uber trip data, identifying key patterns, and predicting fare amounts using Machine Learning. The analysis includes data cleaning, exploratory data analysis (EDA), hypothesis testing, and building a Random Forest regression model.  

---

## 🔹 Key Features  
✔️ **Data Cleaning:** Handling missing values, duplicates, and outliers  
✔️ **Feature Engineering:** Extracting time-based features from trip timestamps  
✔️ **Exploratory Data Analysis (EDA):**  
   - Trips distribution by hour and day of the week  
   - Fare amount distribution  
   - Passenger count vs. fare trends  
✔️ **Hypothesis Testing:**  
   - Peak-hour vs. non-peak fares (T-test)  
   - Passenger count impact on fare (ANOVA)  
   - Distance vs. fare correlation (Pearson Test)  
✔️ **Machine Learning Model:**  
   - **Random Forest Regressor** for fare prediction  
   - Performance evaluation (MAE, MSE, R² Score)  
   - Feature importance analysis  
✔️ **A/B Testing:** Analyzing the impact of a 10% fare increase on pricing  

---

## 🛠️ Technologies Used  
- **Python:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`, `SciPy`  
- **Machine Learning:** `Random Forest Regressor`, `GridSearchCV`  
- **Statistical Tests:** T-test, ANOVA, Pearson Correlation  
- **Data Visualization:** Matplotlib, Seaborn  

---

## ⚙️ Project Workflow  
1️⃣ **Data Preprocessing:** Loading and cleaning the Uber dataset  
2️⃣ **Exploratory Data Analysis (EDA):** Visualizing trends and distributions  
3️⃣ **Feature Engineering:** Extracting useful time-based features  
4️⃣ **Hypothesis Testing:** Validating statistical relationships  
5️⃣ **Machine Learning Model:** Training a Random Forest Regressor  
6️⃣ **Model Evaluation:** Checking performance using MAE, MSE, and R² score  
7️⃣ **A/B Testing:** Simulating a fare increase impact analysis  

---

## 📊 Model Performance  
| Metric  | Value  |
|---------|--------|
| **Mean Absolute Error (MAE)** | 2.08 |
| **Mean Squared Error (MSE)** |  18.13 |
| **R² Score** | 0.81 |

---

## 🔮 Predict Your Fare!  
created a function to predict Uber fares based on trip details:  

def predict_fare(hour, passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude):
    # Function logic here
    return predicted_fare
predicted_price = predict_fare(13, 5, -73.9855, 40.7580, -74.0000, 40.7128)
print(f"Predicted Fare Amount: ${predicted_price}")


## Conclusion & Insights

📊 Peak-hour fares tend to be higher than non-peak fares.

📊 Trip distance is strongly correlated with fare amount.

📊 Passenger count has a minor impact on pricing.

📊 The Random Forest model successfully predicts fares with high accuracy.

📊 A/B testing indicates that a 10% fare increase significantly affects pricing.
