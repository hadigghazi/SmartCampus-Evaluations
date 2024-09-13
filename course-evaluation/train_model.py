import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from preprocess import preprocess_data, load_data
import pandas as pd

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
