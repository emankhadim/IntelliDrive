# IntelliDrive: AI-Powered Driver Behavior Analysis

IntelliDrive is a machine learning project aimed at classifying driving behaviors (Slow, Normal, Aggressive) using accelerometer and gyroscope data. The project leverages data collected from smartphones to predict dangerous driving behavior accurately, enhancing road safety.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Aggressive driving behavior is a significant factor in road traffic accidents. Predicting such behavior quickly and accurately can help in taking preventive measures. Using sensors available in modern smartphones, specifically accelerometers and gyroscopes, this project gathers relevant data to classify driving behaviors.

## Data Description

- **Source**: Kaggle
- **Sampling Rate**: 2 samples per second
- **Sensors**: Accelerometer and Gyroscope
- **Features**:
  - Acceleration (X, Y, Z axis in m/s²)
  - Rotation (X, Y, Z axis in °/s)
- **Target Labels**: 
  - SLOW
  - NORMAL
  - AGGRESSIVE
- **Device Used**: Samsung Galaxy S21

## Technologies Used

- **Programming Language**: Python
- **Libraries**: 
  - Pandas
  - NumPy
  - Scikit-learn
  - Streamlit
  - Plotly
- **Machine Learning Model**: Random Forest Classifier
- **Visualization Tools**: Plotly, Streamlit

## Installation

To get a local copy up and running, follow these steps:

### Prerequisites

Ensure you have Python installed. You can download it from [Python.org](https://www.python.org/downloads/).

### Clone the Repository

```sh
git clone https://github.com/emankhadim/IntelliDrive.git
cd IntelliDrive

#### Install Dependencies:

'''sh 
pip install -r requirements.txt
#### Run the Streamlit Application:

'''sh
streamlit run app.py

#### Use the Application:

Open your web browser and go to the URL provided by Streamlit (usually http://localhost:8501).
Use the sidebar to input real-time sensor data (Acceleration and Gyroscope values).
The application will predict the driving behavior and display the result.
Visualize historical data through the scatter matrix plot provided in the application.

Run the Streamlit Application:

sh
Copy code
streamlit run app.py
