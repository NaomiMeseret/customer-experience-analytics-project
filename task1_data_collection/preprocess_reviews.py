"""
Preprocessing Script for Google Play Store Reviews
Cleans and normalizes review data for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

INPUT_DIR = '../data/raw'
OUTPUT_DIR = '../data/processed'
INPUT_FILE = 'all_reviews_raw.csv'
OUTPUT_FILE = 'reviews_cleaned.csv'


def preprocess_reviews(df):
    """
    Preprocess review data: remove duplicates, handle missing data, normalize dates
    
    Args:
        df: Raw DataFrame with reviews
    
    Returns:
        Cleaned DataFrame
    """
    print(f"Starting preprocessing...")
    print(f"Initial records: {len(df)}")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # 1. Remove duplicates based on review text and bank
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['review', 'bank'], keep='first')
    duplicates_removed = initial_count - len(df_clean)
    print(f"Removed {duplicates_removed} duplicate reviews")
    
    # 2. Handle missing data
    print(f"\nMissing data before cleaning:")
    print(df_clean.isnull().sum())
    
    # Remove rows with missing review text (critical field)
    df_clean = df_clean.dropna(subset=['review'])
    
    # Fill missing dates with 'Unknown' or use a default date
    df_clean['date'] = df_clean['date'].fillna('Unknown')
    
    # Fill missing ratings with median rating for that bank
    for bank in df_clean['bank'].unique():
        bank_median = df_clean[df_clean['bank'] == bank]['rating'].median()
        df_clean.loc[(df_clean['bank'] == bank) & (df_clean['rating'].isna()), 'rating'] = bank_median
    
    # Fill any remaining missing ratings with overall median
    df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].median())
    
    # Ensure rating is integer
    df_clean['rating'] = df_clean['rating'].astype(int)
    
    # Fill missing bank name (shouldn't happen, but just in case)
    df_clean['bank'] = df_clean['bank'].fillna('Unknown')
    
    # Fill missing source
    df_clean['source'] = df_clean['source'].fillna('Google Play Store')
    
    print(f"\nMissing data after cleaning:")
    print(df_clean.isnull().sum())
    
    # 3. Normalize dates
    # Convert date strings to datetime, handle 'Unknown' dates
    def normalize_date(date_str):
        if pd.isna(date_str) or date_str == 'Unknown':
            return 'Unknown'
        try:
            # Try parsing the date
            if isinstance(date_str, str):
                # Handle different date formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
                # If no format matches, try pandas parser
                dt = pd.to_datetime(date_str, errors='coerce')
                if pd.notna(dt):
                    return dt.strftime('%Y-%m-%d')
            return 'Unknown'
        except:
            return 'Unknown'
    
    df_clean['date'] = df_clean['date'].apply(normalize_date)
    
    # 4. Clean review text
    # Remove extra whitespace
    df_clean['review'] = df_clean['review'].str.strip()
    df_clean['review'] = df_clean['review'].str.replace(r'\s+', ' ', regex=True)
    
    # Remove empty reviews (after cleaning)
    df_clean = df_clean[df_clean['review'].str.len() > 0]
    
    # 5. Validate ratings (should be 1-5)
    df_clean = df_clean[(df_clean['rating'] >= 1) & (df_clean['rating'] <= 5)]
    
    # 6. Add review_id for tracking
    df_clean.insert(0, 'review_id', range(1, len(df_clean) + 1))
    
    print(f"\nFinal records: {len(df_clean)}")
    print(f"Records removed: {initial_count - len(df_clean)}")
    
    return df_clean


def main():
    """Main preprocessing function"""
    print("="*60)
    print("Review Data Preprocessing")
    print("="*60)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read raw data
    input_path = os.path.join(INPUT_DIR, INPUT_FILE)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please run scrape_reviews.py first to collect data.")
        return
    
    print(f"Reading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Preprocess
    df_clean = preprocess_reviews(df)
    
    # Save cleaned data
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df_clean.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Cleaned data saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\nData Quality Summary:")
    print(f"Total reviews: {len(df_clean)}")
    print(f"Missing data percentage: {(df_clean.isnull().sum().sum() / (len(df_clean) * len(df_clean.columns))) * 100:.2f}%")
    
    print("\nReviews by Bank:")
    print(df_clean.groupby('bank').size())
    
    print("\nReviews by Rating:")
    print(df_clean.groupby('rating').size())
    
    print("\nSample of cleaned data:")
    print(df_clean.head())


if __name__ == "__main__":
    main()

