import streamlit as st
import numpy as np
import plotly.express as px
from data_preprocessing import load_and_clean_data
from model_training import train_model, evaluate_model

# Load and preprocess data
X_train, y_train_encoded, X_test, label_encoder = load_and_clean_data('train_motion_data.csv', 'test_motion_data.csv')

# Train model
model, y_val_split, y_val_pred = train_model(X_train, y_train_encoded)

# Evaluate model
class_report, conf_matrix = evaluate_model(y_val_split, y_val_pred, label_encoder)
st.text(class_report)
st.text(conf_matrix)

# Streamlit app
st.title("IntelliDrive: AI-Powered Driver Behavior Analysis")

st.sidebar.header("Input Features")
acc_x = st.sidebar.number_input("Acceleration X (m/s^2)", min_value=-10.0, max_value=10.0, value=0.0)
acc_y = st.sidebar.number_input("Acceleration Y (m/s^2)", min_value=-10.0, max_value=10.0, value=0.0)
acc_z = st.sidebar.number_input("Acceleration Z (m/s^2)", min_value=-10.0, max_value=10.0, value=0.0)
gyro_x = st.sidebar.number_input("Gyroscope X (°/s)", min_value=-500.0, max_value=500.0, value=0.0)
gyro_y = st.sidebar.number_input("Gyroscope Y (°/s)", min_value=-500.0, max_value=500.0, value=0.0)
gyro_z = st.sidebar.number_input("Gyroscope Z (°/s)", min_value=-500.0, max_value=500.0, value=0.0)

input_data = np.array([[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]])
prediction = model.predict(input_data)
predicted_class = label_encoder.inverse_transform(prediction)[0]

st.subheader("Predicted Driver Behavior")
st.write(predicted_class)

# Display historical data
train_data = pd.read_csv('train_motion_data.csv')
features = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
target = 'Class'

st.subheader("Historical Data Analysis")
fig = px.scatter_matrix(train_data, dimensions=features, color=target, title="Driving Behavior Data")
st.plotly_chart(fig)
