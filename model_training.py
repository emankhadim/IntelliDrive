from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_model(X_train, y_train_encoded):
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_split, y_train_split)

    y_val_pred = model.predict(X_val_split)
    return model, y_val_split, y_val_pred

def evaluate_model(y_val_split, y_val_pred, label_encoder):
    class_names = label_encoder.classes_.astype(str)
    class_report = classification_report(y_val_split, y_val_pred, target_names=class_names)
    conf_matrix = confusion_matrix(y_val_split, y_val_pred)
    return class_report, conf_matrix
