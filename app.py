import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import time

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Streamlit UI
st.title("🏋️ Personal Fitness Tracker")
st.write("### Predict your calories burned based on exercise parameters")

st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min)", 0, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 200, 90)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35, 42, 37)
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0
    
    features = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender_male": [gender]
    })
    return features

df = user_input_features()

st.write("## Your Selected Parameters")
st.write(df)

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns=["User_ID"], inplace=True)
exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df = pd.get_dummies(exercise_df, drop_first=True)

# Train-test split
X = exercise_df.drop(columns=["Calories"])
y = exercise_df["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Align input features with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Predict calories burned
prediction = linear_reg.predict(df)[0]
st.write(f"## 🔥 Predicted Calories Burned: {round(prediction, 2)} kcal")

# Display Model Performance
st.write("### Model Performance (Linear Regression)")
y_pred = linear_reg.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
st.write(f"Mean Absolute Error: {round(mae, 2)}")
st.write(f"Mean Squared Error: {round(mse, 2)}")

# Additional Analytics
st.write("## 📊 Additional Insights")
st.write("### Exercise & Calorie Trends")
fig, ax = plt.subplots()
sns.scatterplot(x=exercise_df["Duration"], y=exercise_df["Calories"], alpha=0.6, ax=ax)
ax.set_xlabel("Exercise Duration (min)")
ax.set_ylabel("Calories Burned")
st.pyplot(fig)

st.write("### BMI Classification")
bmi_category = "Underweight" if df["BMI"].values[0] < 18.5 else "Normal" if df["BMI"].values[0] < 25 else "Overweight"
st.write(f"Your BMI category is: **{bmi_category}**")

st.write("### Exercise Recommendations")
if prediction < 100:
    st.write("🔹 Try increasing your workout intensity or duration for better calorie burn!")
elif prediction < 300:
    st.write("✅ You're on the right track! Keep up the good work.")
else:
    st.write("🔥 Amazing job! You have a high-calorie burn workout.")
