import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocess import preprocess_data, load_data

def explain_model(model, X, scaler):
    X_scaled = scaler.transform(X) 
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)
    print("SHAP values calculated.")
    
def evaluate_model():
    file_path = 'finalDataset0.2.xlsx'
    df = load_data(file_path)
    X, y = preprocess_data(df)
    
    print("Class distribution before split:", y.value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = joblib.load('course_success_model.pkl')
    scaler = joblib.load('scaler.pkl') 
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    print(classification_report(y_test, y_pred))
    
    explain_model(model, X_train, scaler)

if __name__ == "__main__":
    evaluate_model()
