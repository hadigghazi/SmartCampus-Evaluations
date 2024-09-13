import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_excel(file_path)
    print("Columns in the dataset:", df.columns)
    return df

def preprocess_data(df):
    df['success'] = df[['teaching_number', 'coursecontent_number', 'examination_number',
                        'labwork_number', 'library_facilities_number', 'extracurricular_number']].mean(axis=1)
    
    df['success_category'] = pd.cut(df['success'],
                                    bins=[-np.inf, -0.1, 0.5, np.inf],
                                    labels=['Unsuccessful', 'Satisfactory', 'Successful'])
    df['success_category'] = df['success_category'].replace('Satisfactory', 'Successful')
    df['success_category'] = pd.Categorical(df['success_category'])
    
    weights = {
        'teaching_number': 4,
        'coursecontent_number': 3,
        'examination_number': 2,
        'labwork_number': 1,
        'library_facilities_number': 1,
        'extracurricular_number': 1
    }
    
    for feature, weight in weights.items():
        df[feature] *= weight
    
    features = ['teaching_number', 'coursecontent_number', 'examination_number',
                'labwork_number', 'library_facilities_number', 'extracurricular_number']
    
    X = df[features]
    y = df['success_category']
    return X, y

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = train_model(X_train, y_train)

    print("Model training completed.")

if __name__ == "__main__":
    main()
