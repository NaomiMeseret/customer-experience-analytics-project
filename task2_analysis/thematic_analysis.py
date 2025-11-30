"""
Thematic Analysis Script
Extracts keywords and groups them into themes for each bank
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re
from collections import Counter
import os

# Load spaCy model (should be downloaded: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Please install it:")
    print("python -m spacy download en_core_web_sm")
    nlp = None

INPUT_DIR = '../data/processed'
OUTPUT_DIR = '../data/processed'
INPUT_FILE = 'reviews_with_sentiment.csv'
OUTPUT_FILE = 'reviews_with_themes.csv'

# Theme definitions (keywords that map to themes)
THEME_KEYWORDS = {
    'Account Access Issues': [
        'login', 'password', 'pin', 'authentication', 'access', 'locked', 'blocked',
        'verification', 'security', 'biometric', 'fingerprint', 'face id'
    ],
    'Transaction Performance': [
        'transfer', 'transaction', 'slow', 'fast', 'speed', 'timeout', 'failed',
        'pending', 'delay', 'processing', 'instant', 'quick', 'lag', 'freeze'
    ],
    'User Interface & Experience': [
        'ui', 'interface', 'design', 'layout', 'navigation', 'user friendly',
        'easy', 'confusing', 'cluttered', 'modern', 'outdated', 'button', 'menu',
        'screen', 'display', 'visual', 'aesthetic'
    ],
    'Customer Support': [
        'support', 'help', 'service', 'contact', 'response', 'assistance',
        'complaint', 'issue', 'problem', 'resolve', 'chat', 'call', 'email'
    ],
    'Feature Requests': [
        'feature', 'add', 'missing', 'need', 'want', 'request', 'suggest',
        'improve', 'enhancement', 'new', 'option', 'functionality', 'capability'
    ],
    'App Reliability': [
        'crash', 'error', 'bug', 'glitch', 'freeze', 'hang', 'not working',
        'broken', 'issue', 'problem', 'stable', 'reliable', 'unstable'
    ],
    'Payment & Banking Features': [
        'payment', 'bill', 'mobile money', 'm-pesa', 'telebirr', 'cbe birr',
        'balance', 'account', 'deposit', 'withdraw', 'statement', 'history'
    ]
}


def preprocess_text(text):
    """
    Preprocess text for keyword extraction
    
    Args:
        text: Raw review text
    
    Returns:
        Preprocessed text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_keywords_tfidf(texts, max_features=50):
    """
    Extract keywords using TF-IDF
    
    Args:
        texts: List of review texts
        max_features: Maximum number of keywords to extract
    
    Returns:
        List of top keywords
    """
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Include unigrams and bigrams
        stop_words='english',
        min_df=2  # Word must appear in at least 2 documents
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores
        mean_scores = tfidf_matrix.mean(axis=0).A1
        keyword_scores = list(zip(feature_names, mean_scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        top_keywords = [kw[0] for kw in keyword_scores[:max_features]]
        return top_keywords
    except:
        return []


def extract_keywords_spacy(texts, max_keywords=50):
    """
    Extract keywords using spaCy (if available)
    
    Args:
        texts: List of review texts
        max_keywords: Maximum number of keywords to extract
    
    Returns:
        List of keywords
    """
    if nlp is None:
        return []
    
    all_keywords = []
    
    for text in texts:
        if pd.isna(text) or text == '':
            continue
        
        doc = nlp(str(text).lower())
        
        # Extract nouns, adjectives, and verbs
        keywords = [
            token.lemma_ for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and (token.pos_ in ['NOUN', 'ADJ', 'VERB'])
            and len(token.lemma_) > 2
        ]
        
        all_keywords.extend(keywords)
    
    # Count keyword frequencies
    keyword_counts = Counter(all_keywords)
    
    # Return top keywords
    top_keywords = [kw[0] for kw in keyword_counts.most_common(max_keywords)]
    return top_keywords


def identify_themes(review_text, theme_keywords):
    """
    Identify themes in a review based on keyword matching
    
    Args:
        review_text: Review text
        theme_keywords: Dictionary mapping theme names to keyword lists
    
    Returns:
        List of identified themes
    """
    if pd.isna(review_text):
        return []
    
    review_lower = str(review_text).lower()
    identified_themes = []
    
    for theme, keywords in theme_keywords.items():
        # Check if any keyword appears in the review
        for keyword in keywords:
            if keyword.lower() in review_lower:
                identified_themes.append(theme)
                break  # Only add theme once per review
    
    return identified_themes


def analyze_themes_by_bank(df):
    """
    Analyze themes for each bank and identify top themes
    
    Args:
        df: DataFrame with reviews and themes
    
    Returns:
        Dictionary with theme analysis per bank
    """
    theme_analysis = {}
    
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]
        
        # Count theme occurrences
        all_themes = []
        for themes in bank_df['themes']:
            if isinstance(themes, str):
                # Parse themes from string representation
                themes_list = eval(themes) if themes.startswith('[') else [themes]
                all_themes.extend(themes_list)
        
        theme_counts = Counter(all_themes)
        
        theme_analysis[bank] = {
            'total_reviews': len(bank_df),
            'theme_counts': dict(theme_counts.most_common()),
            'top_themes': [theme for theme, count in theme_counts.most_common(5)]
        }
    
    return theme_analysis


def main():
    """Main thematic analysis function"""
    print("="*60)
    print("Thematic Analysis - Keyword Extraction and Clustering")
    print("="*60)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read data with sentiment
    input_path = os.path.join(INPUT_DIR, INPUT_FILE)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please run sentiment_analysis.py first.")
        return
    
    print(f"Reading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reviews")
    
    # Extract keywords for each bank
    print("\nExtracting keywords by bank...")
    bank_keywords = {}
    
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]
        bank_texts = bank_df['review'].tolist()
        
        print(f"\n{bank}:")
        print(f"  Processing {len(bank_texts)} reviews...")
        
        # Extract keywords using TF-IDF
        tfidf_keywords = extract_keywords_tfidf(bank_texts, max_features=30)
        print(f"  Top TF-IDF keywords: {tfidf_keywords[:10]}")
        
        # Extract keywords using spaCy (if available)
        if nlp:
            spacy_keywords = extract_keywords_spacy(bank_texts, max_keywords=30)
            print(f"  Top spaCy keywords: {spacy_keywords[:10]}")
        
        bank_keywords[bank] = tfidf_keywords
    
    # Identify themes for each review
    print("\nIdentifying themes for each review...")
    themes_list = []
    
    for idx, row in df.iterrows():
        themes = identify_themes(row['review'], THEME_KEYWORDS)
        themes_list.append(themes)
    
    df['themes'] = themes_list
    df['theme_count'] = df['themes'].apply(len)
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Analyze themes by bank
    theme_analysis = analyze_themes_by_bank(df)
    
    print("\n" + "="*60)
    print("Theme Analysis Summary by Bank")
    print("="*60)
    
    for bank, analysis in theme_analysis.items():
        print(f"\n{bank}:")
        print(f"  Total reviews: {analysis['total_reviews']}")
        print(f"  Top 5 themes:")
        for theme, count in list(analysis['theme_counts'].items())[:5]:
            percentage = (count / analysis['total_reviews']) * 100
            print(f"    - {theme}: {count} reviews ({percentage:.1f}%)")
    
    # Overall statistics
    print("\n" + "="*60)
    print("Overall Theme Distribution")
    print("="*60)
    
    all_themes = []
    for themes in df['themes']:
        all_themes.extend(themes)
    
    theme_counts = Counter(all_themes)
    for theme, count in theme_counts.most_common():
        percentage = (count / len(df)) * 100
        print(f"  {theme}: {count} reviews ({percentage:.1f}%)")
    
    # Reviews without themes
    no_theme_count = len(df[df['theme_count'] == 0])
    print(f"\nReviews without identified themes: {no_theme_count} ({no_theme_count/len(df)*100:.1f}%)")
    
    print("\nSample results:")
    sample_df = df[df['theme_count'] > 0][['review_id', 'bank', 'rating', 'themes']].head(10)
    for idx, row in sample_df.iterrows():
        print(f"\nReview {row['review_id']} ({row['bank']}, {row['rating']} stars):")
        print(f"  Themes: {row['themes']}")


if __name__ == "__main__":
    main()

