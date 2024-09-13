import joblib
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data, load_data

def evaluate_model():
    file_path = 'finalDataset0.2.xlsx'
    df = load_data(file_path)
    X, y = preprocess_data(df)
    
    print("Class distribution before split:", y.value_counts())

if __name__ == "__main__":
    evaluate_model()
