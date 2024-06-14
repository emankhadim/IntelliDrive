import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    features = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
    target = 'Class'

    def clean_data(df):
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = pd.to_numeric(df[column], errors='coerce')
        df.fillna(df.mean(), inplace=True)
        return df

    train_data = clean_data(train_data)
    test_data = clean_data(test_data)

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    return X_train, y_train_encoded, X_test, label_encoder
