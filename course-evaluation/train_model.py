import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data, load_data
import pandas as pd

def train_model(X, y):
    model = GradientBoostingClassifier()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    return model

def main():
    file_path = 'finalDataset0.2.xlsx'
    df = load_data(file_path)
    X, y = preprocess_data(df)
    
    print("Class distribution before split:", y.value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = train_model(X_train, y_train)
    print("Model training completed.")

if __name__ == "__main__":
    main()
