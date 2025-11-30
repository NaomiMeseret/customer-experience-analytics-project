"""
Web Scraping Script for Google Play Store Reviews
Scrapes reviews for three Ethiopian banks: CBE, BOA, and Dashen Bank
"""

import pandas as pd
from google_play_scraper import app, reviews, Sort
import time
from datetime import datetime
import os

# Bank app IDs on Google Play Store
# IMPORTANT: Update these with actual app IDs from Google Play Store
# To find app ID:
# 1. Search for the app on Google Play Store
# 2. Open the app page
# 3. The app ID is in the URL: https://play.google.com/store/apps/details?id=APP_ID_HERE
# Example: If URL is https://play.google.com/store/apps/details?id=com.cbe.mobilebanking
#          Then app_id = 'com.cbe.mobilebanking'

BANK_APPS = {
    'CBE': {
        'app_id': 'com.cbe.mobilebanking',  # TODO: Update with actual CBE app ID
        'app_name': 'Commercial Bank of Ethiopia Mobile'
    },
    'BOA': {
        'app_id': 'com.bankofabyssinia.mobilebanking',  # TODO: Update with actual BOA app ID
        'app_name': 'Bank of Abyssinia Mobile'
    },
    'Dashen': {
        'app_id': 'com.dashenbank.mobilebanking',  # TODO: Update with actual Dashen app ID
        'app_name': 'Dashen Bank Mobile'
    }
}

TARGET_REVIEWS_PER_BANK = 400
OUTPUT_DIR = '../data/raw'


def scrape_bank_reviews(bank_name, app_id, app_name, count=TARGET_REVIEWS_PER_BANK):
    """
    Scrape reviews for a specific bank app
    
    Args:
        bank_name: Short name of the bank (e.g., 'CBE')
        app_id: Google Play Store app ID
        app_name: Full app name
        count: Number of reviews to scrape
    
    Returns:
        DataFrame with reviews
    """
    print(f"\n{'='*60}")
    print(f"Scraping reviews for {bank_name} ({app_name})")
    print(f"{'='*60}")
    
    all_reviews = []
    continuation_token = None
    reviews_collected = 0
    
    try:
        # First batch
        result, continuation_token = reviews(
            app_id,
            lang='en',
            country='et',  # Ethiopia
            sort=Sort.NEWEST,
            count=200,  # Maximum per request
            continuation_token=continuation_token
        )
        
        for review in result:
            all_reviews.append({
                'review': review['content'],
                'rating': review['score'],
                'date': review['at'].strftime('%Y-%m-%d') if review['at'] else None,
                'bank': bank_name,
                'source': 'Google Play Store'
            })
            reviews_collected += 1
        
        print(f"Collected {reviews_collected} reviews so far...")
        
      
        while reviews_collected < count and continuation_token:
            time.sleep(2)  
            
            try:
                result, continuation_token = reviews(
                    app_id,
                    lang='en',
                    country='et',
                    sort=Sort.NEWEST,
                    count=200,
                    continuation_token=continuation_token
                )
                
                for review in result:
                    all_reviews.append({
                        'review': review['content'],
                        'rating': review['score'],
                        'date': review['at'].strftime('%Y-%m-%d') if review['at'] else None,
                        'bank': bank_name,
                        'source': 'Google Play Store'
                    })
                    reviews_collected += 1
                
                print(f"Collected {reviews_collected} reviews so far...")
                
                if len(result) == 0:
                    break
                    
            except Exception as e:
                print(f"Error during continuation: {e}")
                break
        
        print(f"✓ Successfully collected {len(all_reviews)} reviews for {bank_name}")
        return pd.DataFrame(all_reviews)
        
    except Exception as e:
        print(f"✗ Error scraping {bank_name}: {e}")
        print(f"Note: You may need to update the app_id for {bank_name}")
        return pd.DataFrame()


def main():
    """Main function to scrape reviews for all banks"""
    print("="*60)
    print("Google Play Store Review Scraper")
    print("="*60)
    
   
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_reviews_df = pd.DataFrame()
    
    # Scrape reviews for each bank
    for bank_name, bank_info in BANK_APPS.items():
        df = scrape_bank_reviews(
            bank_name=bank_name,
            app_id=bank_info['app_id'],
            app_name=bank_info['app_name'],
            count=TARGET_REVIEWS_PER_BANK
        )
        
        if not df.empty:
            # Save individual bank data
            output_file = os.path.join(OUTPUT_DIR, f'{bank_name}_reviews_raw.csv')
            df.to_csv(output_file, index=False)
            print(f"Saved {bank_name} reviews to {output_file}")
            
            # Combine with all reviews
            all_reviews_df = pd.concat([all_reviews_df, df], ignore_index=True)
        
        time.sleep(3)  
    
    # Save combined data
    if not all_reviews_df.empty:
        combined_output = os.path.join(OUTPUT_DIR, 'all_reviews_raw.csv')
        all_reviews_df.to_csv(combined_output, index=False)
        print(f"\n{'='*60}")
        print(f"Total reviews collected: {len(all_reviews_df)}")
        print(f"Combined data saved to: {combined_output}")
        print(f"{'='*60}")
        
     
        print("\nSummary by Bank:")
        print(all_reviews_df.groupby('bank').size())
        print("\nSummary by Rating:")
        print(all_reviews_df.groupby('rating').size())
    else:
        print("\n⚠ Warning: No reviews were collected. Please check app IDs.")


if __name__ == "__main__":
    main()

