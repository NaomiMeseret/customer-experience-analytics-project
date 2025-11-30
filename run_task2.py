"""
Helper script to run Task 2: Sentiment and Thematic Analysis
"""

import subprocess
import sys
import os

def main():
    """Run Task 2 scripts in sequence"""
    print("="*60)
    print("Running Task 2: Sentiment and Thematic Analysis")
    print("="*60)
    
    # Change to task2 directory
    task2_dir = os.path.join(os.path.dirname(__file__), 'task2_analysis')
    os.chdir(task2_dir)
    
    # Run sentiment analysis script
    print("\nStep 1: Analyzing sentiment using DistilBERT...")
    try:
        subprocess.run([sys.executable, 'sentiment_analysis.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running sentiment_analysis.py: {e}")
        return
    
    # Run thematic analysis script
    print("\nStep 2: Extracting themes and keywords...")
    try:
        subprocess.run([sys.executable, 'thematic_analysis.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running thematic_analysis.py: {e}")
        return
    
    print("\n" + "="*60)
    print("Task 2 completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()

