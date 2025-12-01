"""
PostgreSQL Data Insertion Script
Inserts cleaned review data into PostgreSQL database
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os
import sys
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'bank_reviews',
    'user': 'postgres',  # Update with your PostgreSQL username
    'password': '',  # Update with your PostgreSQL password
    'port': 5432
}

DATA_FILE = '../data/processed/reviews_cleaned.csv'
SENTIMENT_FILE = '../data/processed/reviews_with_sentiment.csv'
THEMES_FILE = '../data/processed/reviews_with_themes.csv'


def get_db_connection():
    """Create and return database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Successfully connected to PostgreSQL database")
        return conn
    except psycopg2.Error as e:
        print(f"❌ Error connecting to database: {e}")
        print("\n💡 Make sure PostgreSQL is running and database 'bank_reviews' exists")
        print("💡 Update DB_CONFIG in this script with your credentials")
        sys.exit(1)


def insert_banks(conn):
    """Insert bank data into banks table"""
    cursor = conn.cursor()
    
    banks_data = [
        ('CBE', 'Commercial Bank of Ethiopia Mobile'),
        ('BOA', 'Bank of Abyssinia Mobile'),
        ('Dashen', 'Dashen Bank Mobile')
    ]
    
    try:
        for bank_name, app_name in banks_data:
            cursor.execute(
                "INSERT INTO banks (bank_name, app_name) VALUES (%s, %s) ON CONFLICT (bank_name) DO NOTHING",
                (bank_name, app_name)
            )
        conn.commit()
        print("✅ Bank data inserted successfully")
    except psycopg2.Error as e:
        print(f"❌ Error inserting bank data: {e}")
        conn.rollback()


def get_bank_id_map(conn):
    """Get mapping of bank names to bank_ids"""
    cursor = conn.cursor()
    cursor.execute("SELECT bank_id, bank_name FROM banks")
    return {row[1]: row[0] for row in cursor.fetchall()}


def load_review_data():
    """Load review data from CSV files"""
    # Try to load with sentiment and themes if available
    if os.path.exists(THEMES_FILE):
        df = pd.read_csv(THEMES_FILE)
        print(f"✅ Loaded data with themes: {len(df)} reviews")
    elif os.path.exists(SENTIMENT_FILE):
        df = pd.read_csv(SENTIMENT_FILE)
        print(f"✅ Loaded data with sentiment: {len(df)} reviews")
    elif os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        print(f"✅ Loaded cleaned data: {len(df)} reviews")
    else:
        print(f"❌ No data file found. Expected one of:")
        print(f"   - {THEMES_FILE}")
        print(f"   - {SENTIMENT_FILE}")
        print(f"   - {DATA_FILE}")
        sys.exit(1)
    
    return df


def prepare_review_data(df, bank_id_map):
    """Prepare review data for insertion"""
    reviews_data = []
    
    for _, row in df.iterrows():
        bank_name = row['bank']
        bank_id = bank_id_map.get(bank_name)
        
        if not bank_id:
            print(f"⚠️  Warning: Bank '{bank_name}' not found in database, skipping review {row['review_id']}")
            continue
        
        # Handle date
        review_date = None
        if pd.notna(row.get('date')) and row.get('date') != 'Unknown':
            try:
                review_date = pd.to_datetime(row['date']).date()
            except:
                review_date = None
        
        # Handle sentiment
        sentiment_label = row.get('sentiment_label') if 'sentiment_label' in row else None
        sentiment_score = row.get('sentiment_score') if 'sentiment_score' in row else None
        
        # Handle themes
        themes = None
        if 'themes' in row and pd.notna(row['themes']):
            try:
                # Parse themes if it's a string representation of list
                if isinstance(row['themes'], str):
                    themes = eval(row['themes']) if row['themes'].startswith('[') else [row['themes']]
                else:
                    themes = row['themes']
            except:
                themes = None
        
        reviews_data.append((
            int(row['review_id']),
            bank_id,
            str(row['review']),
            int(row['rating']),
            review_date,
            sentiment_label,
            float(sentiment_score) if pd.notna(sentiment_score) else None,
            row.get('source', 'Google Play Store'),
            themes
        ))
    
    return reviews_data


def insert_reviews(conn, reviews_data):
    """Insert reviews into database"""
    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO reviews (
            review_id, bank_id, review_text, rating, review_date,
            sentiment_label, sentiment_score, source, themes
        ) VALUES %s
        ON CONFLICT (review_id) DO UPDATE SET
            review_text = EXCLUDED.review_text,
            rating = EXCLUDED.rating,
            review_date = EXCLUDED.review_date,
            sentiment_label = EXCLUDED.sentiment_label,
            sentiment_score = EXCLUDED.sentiment_score,
            themes = EXCLUDED.themes
    """
    
    try:
        execute_values(cursor, insert_query, reviews_data)
        conn.commit()
        print(f"✅ Successfully inserted {len(reviews_data)} reviews")
        return len(reviews_data)
    except psycopg2.Error as e:
        print(f"❌ Error inserting reviews: {e}")
        conn.rollback()
        return 0


def verify_data(conn):
    """Run verification queries"""
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("Data Verification")
    print("="*60)
    
    # Count reviews per bank
    cursor.execute("""
        SELECT b.bank_name, COUNT(r.review_id) as review_count
        FROM banks b
        LEFT JOIN reviews r ON b.bank_id = r.bank_id
        GROUP BY b.bank_name
        ORDER BY review_count DESC
    """)
    
    print("\n📊 Reviews per Bank:")
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]} reviews")
    
    # Average rating per bank
    cursor.execute("""
        SELECT b.bank_name, 
               ROUND(AVG(r.rating), 2) as avg_rating,
               COUNT(r.review_id) as review_count
        FROM banks b
        LEFT JOIN reviews r ON b.bank_id = r.bank_id
        GROUP BY b.bank_name
        ORDER BY avg_rating DESC
    """)
    
    print("\n⭐ Average Rating per Bank:")
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]} stars ({row[2]} reviews)")
    
    # Total reviews
    cursor.execute("SELECT COUNT(*) FROM reviews")
    total = cursor.fetchone()[0]
    print(f"\n📈 Total Reviews in Database: {total}")
    
    # Sentiment distribution (if available)
    cursor.execute("""
        SELECT sentiment_label, COUNT(*) as count
        FROM reviews
        WHERE sentiment_label IS NOT NULL
        GROUP BY sentiment_label
        ORDER BY count DESC
    """)
    
    sentiment_data = cursor.fetchall()
    if sentiment_data:
        print("\n💭 Sentiment Distribution:")
        for row in sentiment_data:
            print(f"   {row[0]}: {row[1]} reviews")


def main():
    """Main function"""
    print("="*60)
    print("PostgreSQL Data Insertion Script")
    print("="*60)
    
    # Check if data file exists
    if not any(os.path.exists(f) for f in [DATA_FILE, SENTIMENT_FILE, THEMES_FILE]):
        print(f"❌ No data file found. Please run preprocessing scripts first.")
        return
    
    # Connect to database
    conn = get_db_connection()
    
    try:
        # Insert banks
        insert_banks(conn)
        
        # Get bank ID mapping
        bank_id_map = get_bank_id_map(conn)
        print(f"✅ Found {len(bank_id_map)} banks in database")
        
        # Load review data
        df = load_review_data()
        
        # Prepare data for insertion
        reviews_data = prepare_review_data(df, bank_id_map)
        
        if not reviews_data:
            print("❌ No reviews to insert")
            return
        
        # Insert reviews
        inserted_count = insert_reviews(conn, reviews_data)
        
        # Verify data
        if inserted_count > 0:
            verify_data(conn)
        
        print("\n" + "="*60)
        print("✅ Data insertion completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        print("\n🔌 Database connection closed")


if __name__ == "__main__":
    main()

