"""
Helper script to run Task 1: Data Collection and Preprocessing
"""

import subprocess
import sys
import os

def main():
    """Run Task 1 scripts in sequence"""
    print("="*60)
    print("Running Task 1: Data Collection and Preprocessing")
    print("="*60)
    
    # Change to task1 directory
    task1_dir = os.path.join(os.path.dirname(__file__), 'task1_data_collection')
    os.chdir(task1_dir)
    
    # Run scraping script
    print("\nStep 1: Scraping reviews from Google Play Store...")
    try:
        subprocess.run([sys.executable, 'scrape_reviews.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running scrape_reviews.py: {e}")
        return
    
    # Run preprocessing script
    print("\nStep 2: Preprocessing review data...")
    try:
        subprocess.run([sys.executable, 'preprocess_reviews.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running preprocess_reviews.py: {e}")
        return
    
    print("\n" + "="*60)
    print("Task 1 completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()

