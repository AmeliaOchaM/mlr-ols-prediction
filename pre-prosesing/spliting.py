import pandas as pd

def split_csv_sequential(file_path, train_file="./pre-prosesing/input_data/train.csv", test_file="./pre-prosesing/input_data/test.csv", split_ratio=0.7):
    """
    Split dataset CSV menjadi train dan test secara berurutan
    """
    # Load CSV
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Hitung batas split
    split_index = int(len(df) * split_ratio)

    # Split berurutan (awal → train, sisanya → test)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # Simpan ke file
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"✅ Train set: {train_df.shape}, saved to {train_file}")
    print(f"✅ Test set:  {test_df.shape}, saved to {test_file}")

    return train_df, test_df

if __name__ == "__main__":
    # Ganti path sesuai file CSV kamu
    file_path = "./pre-prosesing/data_cleaned.csv"
    train_df, test_df = split_csv_sequential(file_path)
