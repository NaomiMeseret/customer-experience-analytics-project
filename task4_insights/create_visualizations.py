"""
Create Visualizations for Insights Report
Generates 3-5 key visualizations for the analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

INPUT_DIR = '../data/processed'
OUTPUT_DIR = '../reports/figures'

# Try to load data with themes, fallback to sentiment or cleaned
DATA_FILES = ['reviews_with_themes.csv', 'reviews_with_sentiment.csv', 'reviews_cleaned.csv']


def load_data():
    """Load the most complete dataset available"""
    for filename in DATA_FILES:
        file_path = os.path.join(INPUT_DIR, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✅ Loaded {len(df)} reviews from {filename}")
            return df
    
    raise FileNotFoundError("No data file found. Please run preprocessing scripts first.")


def plot_1_rating_distribution_by_bank(df):
    """Plot 1: Rating Distribution by Bank"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create rating distribution
    rating_by_bank = pd.crosstab(df['bank'], df['rating'])
    
    # Plot stacked bar chart
    rating_by_bank.plot(kind='bar', ax=ax, 
                        color=['#FF6B6B', '#FFA07A', '#FFD700', '#98D8C8', '#6BCB77'],
                        width=0.8)
    
    ax.set_title('Rating Distribution by Bank', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Bank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
    ax.legend(title='Rating', labels=['1★', '2★', '3★', '4★', '5★'], 
              title_fontsize=11, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot1_rating_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: plot1_rating_distribution.png")


def plot_2_sentiment_trends(df):
    """Plot 2: Sentiment Trends by Bank"""
    if 'sentiment_label' not in df.columns:
        print("⚠️  Skipping sentiment plot - sentiment data not available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sentiment distribution
    sentiment_by_bank = pd.crosstab(df['bank'], df['sentiment_label'])
    sentiment_by_bank.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#6BCB77'], width=0.8)
    ax1.set_title('Sentiment Distribution by Bank', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Bank', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Reviews', fontsize=11, fontweight='bold')
    ax1.legend(title='Sentiment', title_fontsize=10, fontsize=9)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Average sentiment score
    if 'sentiment_score' in df.columns:
        avg_sentiment = df.groupby('bank')['sentiment_score'].mean().sort_values(ascending=False)
        colors = ['#6BCB77' if x > 0.5 else '#FF6B6B' for x in avg_sentiment]
        avg_sentiment.plot(kind='bar', ax=ax2, color=colors, width=0.6)
        ax2.set_title('Average Sentiment Score by Bank', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Bank', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Average Sentiment Score', fontsize=11, fontweight='bold')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral (0.5)')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot2_sentiment_trends.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: plot2_sentiment_trends.png")


def plot_3_theme_analysis(df):
    """Plot 3: Theme Analysis by Bank"""
    if 'themes' not in df.columns:
        print("⚠️  Skipping theme plot - theme data not available")
        return
    
    # Extract themes
    theme_data = []
    for _, row in df.iterrows():
        if pd.notna(row['themes']):
            try:
                if isinstance(row['themes'], str):
                    themes = eval(row['themes']) if row['themes'].startswith('[') else [row['themes']]
                else:
                    themes = row['themes']
                for theme in themes:
                    theme_data.append({'bank': row['bank'], 'theme': theme})
            except:
                pass
    
    if not theme_data:
        print("⚠️  No theme data available for plotting")
        return
    
    theme_df = pd.DataFrame(theme_data)
    theme_counts = theme_df.groupby(['bank', 'theme']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    theme_counts.plot(kind='barh', ax=ax, width=0.8, colormap='Set3')
    ax.set_title('Theme Distribution by Bank', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Reviews', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bank', fontsize=12, fontweight='bold')
    ax.legend(title='Theme', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot3_theme_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: plot3_theme_analysis.png")


def plot_4_comparative_analysis(df):
    """Plot 4: Comparative Analysis Dashboard"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Average Rating Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    avg_rating = df.groupby('bank')['rating'].mean().sort_values(ascending=False)
    colors = ['#2E86AB' if x == avg_rating.max() else '#A23B72' if x == avg_rating.min() else '#F18F01' for x in avg_rating]
    avg_rating.plot(kind='bar', ax=ax1, color=colors, width=0.6)
    ax1.set_title('Average Rating by Bank', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Rating', fontsize=11)
    ax1.set_ylim(0, 5)
    ax1.axhline(y=3.0, color='r', linestyle='--', alpha=0.5, label='Threshold (3.0)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Positive vs Negative Reviews
    ax2 = fig.add_subplot(gs[0, 1])
    positive_pct = df.groupby('bank').apply(lambda x: len(x[x['rating'] >= 4]) / len(x) * 100)
    negative_pct = df.groupby('bank').apply(lambda x: len(x[x['rating'] <= 2]) / len(x) * 100)
    
    x = np.arange(len(positive_pct))
    width = 0.35
    ax2.bar(x - width/2, positive_pct, width, label='Positive (4-5★)', color='#6BCB77')
    ax2.bar(x + width/2, negative_pct, width, label='Negative (1-2★)', color='#FF6B6B')
    ax2.set_title('Positive vs Negative Reviews', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(positive_pct.index, rotation=0)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Review Count
    ax3 = fig.add_subplot(gs[1, 0])
    review_counts = df['bank'].value_counts()
    review_counts.plot(kind='bar', ax=ax3, color='#4ECDC4', width=0.6)
    ax3.set_title('Total Reviews by Bank', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Number of Reviews', fontsize=11)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Rating Distribution (Box Plot)
    ax4 = fig.add_subplot(gs[1, 1])
    banks = df['bank'].unique()
    data_for_box = [df[df['bank'] == bank]['rating'].values for bank in banks]
    bp = ax4.boxplot(data_for_box, labels=banks, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#FFD93D')
    ax4.set_title('Rating Distribution (Box Plot)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Rating', fontsize=11)
    ax4.set_ylim(0.5, 5.5)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Bank Comparison Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot4_comparative_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: plot4_comparative_analysis.png")


def plot_5_keyword_cloud(df):
    """Plot 5: Keyword Cloud (Word Cloud)"""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("⚠️  WordCloud not available. Install with: pip install wordcloud")
        return
    
    # Combine all review text
    all_text = ' '.join(df['review'].astype(str).str.lower())
    
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
                 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'app', 'bank', 'banking'}
    
    # Generate word cloud
    wordcloud = WordCloud(width=1200, height=600, 
                         background_color='white',
                         stopwords=stopwords,
                         max_words=100,
                         colormap='viridis').generate(all_text)
    
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Review Keyword Cloud', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot5_keyword_cloud.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created: plot5_keyword_cloud.png")


def main():
    """Main function"""
    print("="*60)
    print("Creating Visualizations")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Create visualizations
    plot_1_rating_distribution_by_bank(df)
    plot_2_sentiment_trends(df)
    plot_3_theme_analysis(df)
    plot_4_comparative_analysis(df)
    plot_5_keyword_cloud(df)
    
    print("\n" + "="*60)
    print("✅ All visualizations created successfully!")
    print(f"📁 Saved to: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()

