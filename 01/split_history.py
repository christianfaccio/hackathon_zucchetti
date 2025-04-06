import pandas as pd
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python split_history.py <input_file> <train_csv> <test_csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    train_csv = sys.argv[2]
    test_csv = sys.argv[3]
    
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Calculate split index for 90:10 ratio (static split, no shuffling)
    split_index = int(len(df) * 0.9)
    
    # Create train and test DataFrames
    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]
    
    # Save the output CSV files
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    
    print(f"Data split into {len(df_train)} training rows and {len(df_test)} testing rows.")

if __name__ == '__main__':
    main()