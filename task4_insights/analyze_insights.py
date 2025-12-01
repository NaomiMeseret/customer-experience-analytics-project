"""
Insights and Recommendations Analysis
Identifies drivers, pain points, and generates recommendations per bank
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import os

INPUT_DIR = '../data/processed'
OUTPUT_DIR = '../data/processed'
INPUT_FILE = 'reviews_with_themes.csv'
OUTPUT_FILE = 'insights_and_recommendations.md'


def load_data():
    """Load processed review data"""
    file_path = os.path.join(INPUT_DIR, INPUT_FILE)
    
    if not os.path.exists(file_path):
        # Try alternative files
        alt_files = ['reviews_with_sentiment.csv', 'reviews_cleaned.csv']
        for alt_file in alt_files:
            alt_path = os.path.join(INPUT_DIR, alt_file)
            if os.path.exists(alt_path):
                file_path = alt_path
                break
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found. Expected: {INPUT_FILE}")
    
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} reviews")
    return df


def extract_keywords_from_reviews(df, sentiment_filter=None, rating_filter=None):
    """Extract keywords from reviews based on filters"""
    reviews = df.copy()
    
    if sentiment_filter:
        reviews = reviews[reviews['sentiment_label'] == sentiment_filter]
    
    if rating_filter:
        if isinstance(rating_filter, tuple):
            reviews = reviews[reviews['rating'].between(rating_filter[0], rating_filter[1])]
        else:
            reviews = reviews[reviews['rating'] == rating_filter]
    
    # Common positive keywords
    positive_keywords = [
        'fast', 'quick', 'easy', 'good', 'great', 'excellent', 'love', 'best',
        'smooth', 'reliable', 'convenient', 'user friendly', 'simple', 'helpful',
        'secure', 'stable', 'efficient', 'amazing', 'wonderful', 'perfect'
    ]
    
    # Common negative keywords
    negative_keywords = [
        'slow', 'crash', 'error', 'bug', 'problem', 'issue', 'bad', 'terrible',
        'frustrating', 'difficult', 'confusing', 'broken', 'failed', 'poor',
        'unreliable', 'freeze', 'hang', 'timeout', 'disappointed', 'worst'
    ]
    
    # Extract keywords from reviews
    all_text = ' '.join(reviews['review'].astype(str).str.lower())
    
    positive_counts = {kw: all_text.count(kw) for kw in positive_keywords}
    negative_counts = {kw: all_text.count(kw) for kw in negative_keywords}
    
    return positive_counts, negative_counts


def identify_drivers(df, bank_name):
    """Identify satisfaction drivers for a bank"""
    bank_df = df[df['bank'] == bank_name]
    
    # Filter positive reviews (4-5 stars or positive sentiment)
    positive_reviews = bank_df[
        (bank_df['rating'] >= 4) | 
        (bank_df.get('sentiment_label', pd.Series()) == 'POSITIVE')
    ]
    
    if len(positive_reviews) == 0:
        return []
    
    # Extract themes if available
    drivers = []
    
    if 'themes' in positive_reviews.columns:
        all_themes = []
        for themes in positive_reviews['themes']:
            if pd.notna(themes):
                try:
                    if isinstance(themes, str):
                        theme_list = eval(themes) if themes.startswith('[') else [themes]
                    else:
                        theme_list = themes
                    all_themes.extend(theme_list)
                except:
                    pass
        
        theme_counts = Counter(all_themes)
        drivers.extend([theme for theme, count in theme_counts.most_common(3)])
    
    # Extract keywords from positive reviews
    pos_keywords, _ = extract_keywords_from_reviews(positive_reviews)
    top_positive = sorted(pos_keywords.items(), key=lambda x: x[1], reverse=True)[:3]
    drivers.extend([kw for kw, count in top_positive if count > 0])
    
    # Get example reviews
    example_reviews = positive_reviews.nlargest(3, 'rating')['review'].tolist()
    
    return {
        'drivers': list(set(drivers))[:5],
        'examples': example_reviews[:2]
    }


def identify_pain_points(df, bank_name):
    """Identify pain points for a bank"""
    bank_df = df[df['bank'] == bank_name]
    
    # Filter negative reviews (1-2 stars or negative sentiment)
    negative_reviews = bank_df[
        (bank_df['rating'] <= 2) | 
        (bank_df.get('sentiment_label', pd.Series()) == 'NEGATIVE')
    ]
    
    if len(negative_reviews) == 0:
        return []
    
    # Extract themes if available
    pain_points = []
    
    if 'themes' in negative_reviews.columns:
        all_themes = []
        for themes in negative_reviews['themes']:
            if pd.notna(themes):
                try:
                    if isinstance(themes, str):
                        theme_list = eval(themes) if themes.startswith('[') else [themes]
                    else:
                        theme_list = themes
                    all_themes.extend(theme_list)
                except:
                    pass
        
        theme_counts = Counter(all_themes)
        pain_points.extend([theme for theme, count in theme_counts.most_common(3)])
    
    # Extract keywords from negative reviews
    _, neg_keywords = extract_keywords_from_reviews(negative_reviews)
    top_negative = sorted(neg_keywords.items(), key=lambda x: x[1], reverse=True)[:3]
    pain_points.extend([kw for kw, count in top_negative if count > 0])
    
    # Get example reviews
    example_reviews = negative_reviews.nsmallest(3, 'rating')['review'].tolist()
    
    return {
        'pain_points': list(set(pain_points))[:5],
        'examples': example_reviews[:2]
    }


def generate_recommendations(drivers, pain_points, bank_name, df):
    """Generate recommendations based on drivers and pain points"""
    bank_df = df[df['bank'] == bank_name]
    avg_rating = bank_df['rating'].mean()
    
    recommendations = []
    
    # Recommendations based on pain points
    if 'slow' in str(pain_points).lower() or 'Transaction Performance' in str(pain_points):
        recommendations.append({
            'priority': 'High',
            'recommendation': 'Optimize transaction processing speed',
            'rationale': 'Users frequently report slow transaction processing. Consider server optimization, caching, and code refactoring.',
            'impact': 'High - directly addresses user complaints'
        })
    
    if 'crash' in str(pain_points).lower() or 'App Reliability' in str(pain_points):
        recommendations.append({
            'priority': 'High',
            'recommendation': 'Improve app stability and fix crashes',
            'rationale': 'App crashes are a major pain point. Implement comprehensive error handling and testing.',
            'impact': 'High - critical for user retention'
        })
    
    if 'login' in str(pain_points).lower() or 'Account Access Issues' in str(pain_points):
        recommendations.append({
            'priority': 'Medium',
            'recommendation': 'Enhance authentication system',
            'rationale': 'Users experience login difficulties. Consider biometric authentication and password recovery improvements.',
            'impact': 'Medium - improves user experience'
        })
    
    if 'support' in str(pain_points).lower() or 'Customer Support' in str(pain_points):
        recommendations.append({
            'priority': 'Medium',
            'recommendation': 'Improve customer support responsiveness',
            'rationale': 'Users report poor support experience. Consider AI chatbot integration and faster response times.',
            'impact': 'Medium - enhances customer satisfaction'
        })
    
    # Feature recommendations
    if avg_rating < 3.5:
        recommendations.append({
            'priority': 'High',
            'recommendation': 'Conduct comprehensive UX audit',
            'rationale': f'Average rating of {avg_rating:.1f} indicates significant user dissatisfaction. Prioritize user experience improvements.',
            'impact': 'High - addresses root cause of low ratings'
        })
    
    # Positive reinforcement
    if 'fast' in str(drivers).lower() or 'Transaction Performance' in str(drivers):
        recommendations.append({
            'priority': 'Low',
            'recommendation': 'Maintain and enhance transaction speed',
            'rationale': 'Users appreciate fast transactions. Continue optimizing this strength.',
            'impact': 'Low - maintains competitive advantage'
        })
    
    return recommendations


def compare_banks(df):
    """Compare banks across key metrics"""
    comparison = {}
    
    for bank in df['bank'].unique():
        bank_df = df[df['bank'] == bank]
        
        comparison[bank] = {
            'total_reviews': len(bank_df),
            'avg_rating': bank_df['rating'].mean(),
            'positive_pct': len(bank_df[bank_df['rating'] >= 4]) / len(bank_df) * 100,
            'negative_pct': len(bank_df[bank_df['rating'] <= 2]) / len(bank_df) * 100,
        }
        
        if 'sentiment_label' in bank_df.columns:
            sentiment_counts = bank_df['sentiment_label'].value_counts()
            comparison[bank]['positive_sentiment'] = sentiment_counts.get('POSITIVE', 0) / len(bank_df) * 100
            comparison[bank]['negative_sentiment'] = sentiment_counts.get('NEGATIVE', 0) / len(bank_df) * 100
    
    return comparison


def generate_report(df):
    """Generate comprehensive insights report"""
    report = []
    report.append("# Insights and Recommendations Report\n")
    report.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Total Reviews Analyzed**: {len(df)}\n\n")
    
    # Bank comparison
    report.append("## Bank Comparison\n\n")
    comparison = compare_banks(df)
    
    report.append("| Bank | Avg Rating | Positive Reviews (4-5★) | Negative Reviews (1-2★) |\n")
    report.append("|------|------------|-------------------------|-------------------------|\n")
    
    for bank, metrics in sorted(comparison.items(), key=lambda x: x[1]['avg_rating'], reverse=True):
        report.append(f"| {bank} | {metrics['avg_rating']:.2f} | {metrics['positive_pct']:.1f}% | {metrics['negative_pct']:.1f}% |\n")
    
    report.append("\n")
    
    # Per-bank analysis
    for bank in df['bank'].unique():
        report.append(f"## {bank} Analysis\n\n")
        
        # Drivers
        drivers_data = identify_drivers(df, bank)
        report.append("### ✅ Satisfaction Drivers\n\n")
        if drivers_data['drivers']:
            for i, driver in enumerate(drivers_data['drivers'], 1):
                report.append(f"{i}. **{driver}**\n")
            if drivers_data['examples']:
                report.append(f"\n*Example Review*: \"{drivers_data['examples'][0]}\"\n\n")
        else:
            report.append("No clear drivers identified from available data.\n\n")
        
        # Pain points
        pain_points_data = identify_pain_points(df, bank)
        report.append("### ❌ Pain Points\n\n")
        if pain_points_data['pain_points']:
            for i, pain_point in enumerate(pain_points_data['pain_points'], 1):
                report.append(f"{i}. **{pain_point}**\n")
            if pain_points_data['examples']:
                report.append(f"\n*Example Review*: \"{pain_points_data['examples'][0]}\"\n\n")
        else:
            report.append("No clear pain points identified from available data.\n\n")
        
        # Recommendations
        recommendations = generate_recommendations(
            drivers_data['drivers'], 
            pain_points_data['pain_points'], 
            bank, 
            df
        )
        
        report.append("### 💡 Recommendations\n\n")
        if recommendations:
            for rec in recommendations:
                report.append(f"**Priority: {rec['priority']}** - {rec['recommendation']}\n")
                report.append(f"- *Rationale*: {rec['rationale']}\n")
                report.append(f"- *Expected Impact*: {rec['impact']}\n\n")
        else:
            report.append("Continue monitoring user feedback for emerging issues.\n\n")
        
        report.append("---\n\n")
    
    # Ethics and biases
    report.append("## ⚠️ Ethical Considerations and Potential Biases\n\n")
    report.append("### Review Bias Considerations:\n\n")
    report.append("1. **Negative Bias**: Users with negative experiences are more likely to leave reviews than satisfied users, potentially skewing sentiment analysis.\n\n")
    report.append("2. **Selection Bias**: Only users who download and use the app can leave reviews, excluding potential users who chose not to download.\n\n")
    report.append("3. **Recency Bias**: Recent negative experiences may be overrepresented if users are more likely to review immediately after issues.\n\n")
    report.append("4. **Language Bias**: Analysis focuses on English reviews, potentially missing feedback from users who prefer other languages.\n\n")
    report.append("5. **Platform Bias**: Google Play Store reviews may not represent the full user base, especially if users prefer other platforms.\n\n")
    report.append("6. **Cultural Context**: Reviews from Ethiopian users may have cultural nuances that affect sentiment interpretation.\n\n")
    
    report.append("### Recommendations for Mitigation:\n\n")
    report.append("- Consider multiple data sources (App Store, surveys, support tickets)\n")
    report.append("- Weight recent reviews appropriately\n")
    report.append("- Include multi-language support in future analysis\n")
    report.append("- Validate findings with direct user research\n\n")
    
    return ''.join(report)


def main():
    """Main function"""
    print("="*60)
    print("Insights and Recommendations Analysis")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Generate report
    report = generate_report(df)
    
    # Save report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Insights report saved to: {output_path}")
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

