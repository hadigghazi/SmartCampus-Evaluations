import joblib
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data, load_data

def evaluate_model():
    file_path = 'finalDataset0.2.xlsx'
    df = load_data(file_path)
    X, y = preprocess_data(df)
    
    print("Class distribution before split:", y.value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = joblib.load('course_success_model.pkl')
    scaler = joblib.load('scaler.pkl') 
    
    print("Model and scaler loaded successfully.")

if __name__ == "__main__":
    evaluate_model()
