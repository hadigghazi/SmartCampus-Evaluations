import pandas as pd

def load_data(file_path):
    df = pd.read_excel(file_path)
    print("Columns in the dataset:", df.columns)
    return df

def main():
    file_path = 'finalDataset0.2.xlsx'
    df = load_data(file_path)

if __name__ == "__main__":
    main()
