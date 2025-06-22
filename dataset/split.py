import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_dataset(csv_file, output_folder, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15):
    # Baca dataset
    df = pd.read_csv(csv_file)
    
    # Pastikan rasio total = 1
    assert train_ratio + test_ratio + val_ratio == 1.0, "Total rasio harus = 1.0"
    
    print(f"Dataset asli: {len(df)} baris")
    print(f"Distribusi label:")
    print(df['label'].value_counts())
    
    # Split pertama: train vs (test + val)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(test_ratio + val_ratio), 
        stratify=df['label'],  # stratified sampling
        random_state=42
    )
    
    # Split kedua: test vs val
    test_df, val_df = train_test_split(
        temp_df, 
        test_size=val_ratio/(test_ratio + val_ratio), 
        stratify=temp_df['label'],
        random_state=42
    )
    
    # Buat folder output jika belum ada
    os.makedirs(output_folder, exist_ok=True)
    
    # Simpan file
    train_path = os.path.join(output_folder, 'train.csv')
    test_path = os.path.join(output_folder, 'test.csv')
    val_path = os.path.join(output_folder, 'validation.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    # Print hasil
    print(f"\n=== HASIL SPLIT ===")
    print(f"Train: {len(train_df)} baris ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} baris ({len(test_df)/len(df)*100:.1f}%)")
    print(f"Validation: {len(val_df)} baris ({len(val_df)/len(df)*100:.1f}%)")
    
    print(f"\n=== DISTRIBUSI LABEL ===")
    print("Train:")
    print(train_df['label'].value_counts())
    print("\nTest:")
    print(test_df['label'].value_counts())
    print("\nValidation:")
    print(val_df['label'].value_counts())
    
    print(f"\n=== FILE TERSIMPAN ===")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")
    print(f"Validation: {val_path}")
    
    return train_df, test_df, val_df

# Split dataset dengan rasio 70:15:15
train_df, test_df, val_df = split_dataset('dataset.csv', 'splitted')