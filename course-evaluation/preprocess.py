import pandas as pd
import numpy as np

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

    return df

def main():
    file_path = 'finalDataset0.2.xlsx'
    df = load_data(file_path)
    df = preprocess_data(df)
    print("Preprocessed data with weights:", df.head())

if __name__ == "__main__":
    main()
