"""
Sentiment Analysis Script
Uses DistilBERT model to analyze sentiment of reviews
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import os

# Model configuration
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
INPUT_DIR = '../data/processed'
OUTPUT_DIR = '../data/processed'
INPUT_FILE = 'reviews_cleaned.csv'
OUTPUT_FILE = 'reviews_with_sentiment.csv'

# Batch size for processing (adjust based on available memory)
BATCH_SIZE = 32
MAX_LENGTH = 512


class SentimentAnalyzer:
    """Sentiment analyzer using DistilBERT model"""
    
    def __init__(self, model_name=MODEL_NAME):
        """Initialize the sentiment analyzer with the model"""
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Map model outputs to labels
        self.id2label = self.model.config.id2label
        print(f"Model loaded successfully. Labels: {self.id2label}")
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text: Review text
        
        Returns:
            dict with label and score
        """
        if pd.isna(text) or text == '':
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get predicted class and score
        predicted_id = logits.argmax().item()
        predicted_label = self.id2label[predicted_id]
        score = probabilities[0][predicted_id].item()
        
        return {
            'label': predicted_label,
            'score': score
        }
    
    def predict_batch(self, texts):
        """
        Predict sentiment for a batch of texts (more efficient)
        
        Args:
            texts: List of review texts
        
        Returns:
            List of dicts with labels and scores
        """
        # Filter out empty texts
        valid_texts = [text if pd.notna(text) and text != '' else ' ' for text in texts]
        
        # Tokenize batch
        inputs = self.tokenizer(
            valid_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Process results
        results = []
        for i in range(len(texts)):
            predicted_id = logits[i].argmax().item()
            predicted_label = self.id2label[predicted_id]
            score = probabilities[i][predicted_id].item()
            
            # Map to standard labels (POSITIVE, NEGATIVE, NEUTRAL)
            if predicted_label == 'POSITIVE':
                label = 'POSITIVE'
            elif predicted_label == 'NEGATIVE':
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
            
            results.append({
                'label': label,
                'score': score
            })
        
        return results


def analyze_sentiment(df, analyzer):
    """
    Analyze sentiment for all reviews in the dataframe
    
    Args:
        df: DataFrame with reviews
        analyzer: SentimentAnalyzer instance
    
    Returns:
        DataFrame with sentiment columns added
    """
    print(f"\nAnalyzing sentiment for {len(df)} reviews...")
    
    # Prepare results lists
    sentiment_labels = []
    sentiment_scores = []
    
    # Process in batches
    texts = df['review'].tolist()
    num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing batches"):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_results = analyzer.predict_batch(batch_texts)
        
        for result in batch_results:
            sentiment_labels.append(result['label'])
            sentiment_scores.append(result['score'])
    
    # Add sentiment columns to dataframe
    df['sentiment_label'] = sentiment_labels
    df['sentiment_score'] = sentiment_scores
    
    # Map to neutral if score is close to 0.5 (optional)
    # For DistilBERT SST-2, we'll keep POSITIVE/NEGATIVE as is
    
    return df


def aggregate_sentiment_by_bank(df):
    """Aggregate sentiment statistics by bank"""
    print("\n" + "="*60)
    print("Sentiment Analysis Summary by Bank")
    print("="*60)
    
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]
        print(f"\n{bank}:")
        print(f"  Total reviews: {len(bank_df)}")
        print(f"  Positive: {len(bank_df[bank_df['sentiment_label'] == 'POSITIVE'])} ({len(bank_df[bank_df['sentiment_label'] == 'POSITIVE'])/len(bank_df)*100:.1f}%)")
        print(f"  Negative: {len(bank_df[bank_df['sentiment_label'] == 'NEGATIVE'])} ({len(bank_df[bank_df['sentiment_label'] == 'NEGATIVE'])/len(bank_df)*100:.1f}%)")
        print(f"  Average sentiment score: {bank_df['sentiment_score'].mean():.3f}")
        
        # By rating
        print(f"\n  Sentiment by Rating:")
        for rating in sorted(bank_df['rating'].unique()):
            rating_df = bank_df[bank_df['rating'] == rating]
            pos_pct = len(rating_df[rating_df['sentiment_label'] == 'POSITIVE']) / len(rating_df) * 100
            print(f"    {rating} stars: {pos_pct:.1f}% positive, avg score: {rating_df['sentiment_score'].mean():.3f}")


def main():
    """Main sentiment analysis function"""
    print("="*60)
    print("Sentiment Analysis using DistilBERT")
    print("="*60)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read cleaned data
    input_path = os.path.join(INPUT_DIR, INPUT_FILE)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please run preprocess_reviews.py first to clean the data.")
        return
    
    print(f"Reading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reviews")
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze sentiment
    df_with_sentiment = analyze_sentiment(df, analyzer)
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df_with_sentiment.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary statistics
    aggregate_sentiment_by_bank(df_with_sentiment)
    
    # Overall statistics
    print("\n" + "="*60)
    print("Overall Sentiment Distribution")
    print("="*60)
    print(df_with_sentiment['sentiment_label'].value_counts())
    print(f"\nAverage sentiment score: {df_with_sentiment['sentiment_score'].mean():.3f}")
    
    # Coverage check
    coverage = (len(df_with_sentiment[df_with_sentiment['sentiment_label'].notna()]) / len(df_with_sentiment)) * 100
    print(f"\nSentiment coverage: {coverage:.1f}%")
    
    print("\nSample results:")
    print(df_with_sentiment[['review_id', 'bank', 'rating', 'sentiment_label', 'sentiment_score']].head(10))


if __name__ == "__main__":
    main()

