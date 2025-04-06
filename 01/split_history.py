import pandas as pd
import os

def split_data_by_year(input_csv_path, train_csv_path, test_csv_path, test_year=2023):
    """
    Reads a CSV file, splits it into training and test sets based on the year
    extracted from the 'Month' column, and saves them to separate CSV files.

    Args:
        input_csv_path (str): Path to the input CSV file.
        train_csv_path (str): Path where the training CSV file will be saved.
        test_csv_path (str): Path where the test CSV file will be saved.
        test_year (int): The year to use for the test set.
    """
    try:
        # Read the input CSV
        df = pd.read_csv(input_csv_path)
        print(f"Read {len(df)} rows from {input_csv_path}")

        # Ensure 'Month' column exists
        if 'Month' not in df.columns:
            print(f"Error: 'Month' column not found in {input_csv_path}")
            return

        # Extract year from 'Month' column (format like 'Jan2004')
        # Handle potential errors during extraction
        try:
            # Extract the last 4 characters as the year string
            df['Year'] = df['Month'].str[-4:].astype(int)
        except ValueError:
            print("Error: Could not convert extracted year to integer. Check 'Month' column format.")
            # Attempt using pd.to_datetime as an alternative robust method
            try:
                print("Attempting conversion using pd.to_datetime...")
                # errors='coerce' will turn unparseable dates into NaT
                df['parsed_date'] = pd.to_datetime(df['Month'], format='%b%Y', errors='coerce')
                df.dropna(subset=['parsed_date'], inplace=True) # Drop rows that couldn't be parsed
                df['Year'] = df['parsed_date'].dt.year
                df.drop(columns=['parsed_date'], inplace=True) # Remove temporary column
                print("Successfully extracted year using pd.to_datetime.")
            except Exception as e:
                 print(f"Error: Failed to extract year using pd.to_datetime as well. Error: {e}")
                 print("Please ensure the 'Month' column format is consistently like 'MonYYYY' (e.g., 'Jan2023').")
                 return
        except Exception as e:
            print(f"An unexpected error occurred during year extraction: {e}")
            return


        # Split data based on the test year
        test_df = df[df['Year'] == test_year].copy()
        train_df = df[df['Year'] < test_year].copy()

        # Drop the temporary 'Year' column if it exists before saving
        if 'Year' in train_df.columns:
             train_df.drop(columns=['Year'], inplace=True)
        if 'Year' in test_df.columns:
            test_df.drop(columns=['Year'], inplace=True)


        # Ensure output directories exist (optional, creates if not present)
        train_dir = os.path.dirname(train_csv_path)
        if train_dir:
            os.makedirs(train_dir, exist_ok=True)
        test_dir = os.path.dirname(test_csv_path)
        if test_dir:
            os.makedirs(test_dir, exist_ok=True)

        # Save the dataframes to CSV
        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)

        print(f"Training data ({len(train_df)} rows, years < {test_year}) saved to {train_csv_path}")
        print(f"Test data ({len(test_df)} rows, year = {test_year}) saved to {test_csv_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
    except KeyError as e:
         print(f"Error: Missing expected column in {input_csv_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Configuration ---
# Assuming the input file is in the same directory as the script,
# or provide a full/relative path.
INPUT_FILE = 'input/01_input_history.csv'
TRAIN_OUTPUT_FILE = 'train.csv'
TEST_OUTPUT_FILE = 'test.csv'
YEAR_FOR_TEST_SET = 2023
# --- End Configuration ---

if __name__ == "__main__":
    split_data_by_year(INPUT_FILE, TRAIN_OUTPUT_FILE, TEST_OUTPUT_FILE, YEAR_FOR_TEST_SET)