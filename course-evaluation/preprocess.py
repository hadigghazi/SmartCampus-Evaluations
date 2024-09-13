import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import joblib
import shap

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
    
    df = df.dropna()
    X = df[features]
    y = df['success_category']
    
    return X, y

def train_model(X, y):
    model = GradientBoostingClassifier()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    print("Class distribution after SMOTE:", pd.Series(y_resampled).value_counts())
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_resampled, y_resampled)
    
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_}')
    
    best_model = grid_search.best_estimator_
    
    joblib.dump(best_model, 'course_success_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return best_model

def explain_model(model, X, scaler):
    X_scaled = scaler.transform(X) 
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)
    shap.summary_plot(shap_values, X_scaled, feature_names=X.columns)

def main():
    file_path = 'finalDataset0.2.xlsx'
    df = load_data(file_path)
    X, y = preprocess_data(df)
    
    print("Class distribution before split:", y.value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if len(y_train.unique()) > 1:
        model = train_model(X_train, y_train)
        
        scaler = joblib.load('scaler.pkl') 
        X_test_scaled = scaler.transform(X_test)
        
        y_pred = model.predict(X_test_scaled)
        print(classification_report(y_test, y_pred))
        
        explain_model(model, X_train, scaler)
    else:
        print("Not enough classes in training data.")
    
    print("Class distribution in the dataset after preprocessing:", df['success_category'].value_counts())

if __name__ == "__main__":
    main()
