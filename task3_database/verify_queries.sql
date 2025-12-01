-- SQL Verification Queries
-- Run these queries to verify data integrity and get insights

-- 1. Count reviews per bank
SELECT 
    b.bank_name, 
    COUNT(r.review_id) as review_count,
    ROUND(COUNT(r.review_id) * 100.0 / (SELECT COUNT(*) FROM reviews), 2) as percentage
FROM banks b
LEFT JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name
ORDER BY review_count DESC;

-- 2. Average rating per bank
SELECT 
    b.bank_name, 
    ROUND(AVG(r.rating), 2) as avg_rating,
    COUNT(r.review_id) as review_count,
    MIN(r.rating) as min_rating,
    MAX(r.rating) as max_rating
FROM banks b
LEFT JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name
ORDER BY avg_rating DESC;

-- 3. Rating distribution per bank
SELECT 
    b.bank_name,
    r.rating,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY b.bank_name), 2) as percentage
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name, r.rating
ORDER BY b.bank_name, r.rating DESC;

-- 4. Sentiment distribution per bank
SELECT 
    b.bank_name, 
    r.sentiment_label,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY b.bank_name), 2) as percentage,
    ROUND(AVG(r.sentiment_score), 3) as avg_sentiment_score
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
WHERE r.sentiment_label IS NOT NULL
GROUP BY b.bank_name, r.sentiment_label
ORDER BY b.bank_name, r.sentiment_label;

-- 5. Reviews with themes
SELECT 
    b.bank_name,
    UNNEST(r.themes) as theme,
    COUNT(*) as count
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
WHERE r.themes IS NOT NULL AND array_length(r.themes, 1) > 0
GROUP BY b.bank_name, theme
ORDER BY b.bank_name, count DESC;

-- 6. Data quality check - missing data
SELECT 
    'Total Reviews' as metric,
    COUNT(*) as count
FROM reviews
UNION ALL
SELECT 
    'Reviews with Sentiment' as metric,
    COUNT(*) as count
FROM reviews
WHERE sentiment_label IS NOT NULL
UNION ALL
SELECT 
    'Reviews with Themes' as metric,
    COUNT(*) as count
FROM reviews
WHERE themes IS NOT NULL AND array_length(themes, 1) > 0
UNION ALL
SELECT 
    'Reviews with Date' as metric,
    COUNT(*) as count
FROM reviews
WHERE review_date IS NOT NULL;

-- 7. Recent reviews (last 30 days)
SELECT 
    b.bank_name,
    COUNT(*) as recent_reviews,
    ROUND(AVG(r.rating), 2) as avg_rating
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
WHERE r.review_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY b.bank_name
ORDER BY recent_reviews DESC;

-- 8. Top negative reviews (1-2 stars) per bank
SELECT 
    b.bank_name,
    r.rating,
    r.review_text,
    r.sentiment_label,
    r.review_date
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
WHERE r.rating <= 2
ORDER BY b.bank_name, r.rating, r.review_date DESC
LIMIT 10;

-- 9. Top positive reviews (5 stars) per bank
SELECT 
    b.bank_name,
    r.rating,
    r.review_text,
    r.sentiment_label,
    r.review_date
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
WHERE r.rating = 5
ORDER BY b.bank_name, r.review_date DESC
LIMIT 10;

-- 10. Overall statistics
SELECT 
    COUNT(DISTINCT b.bank_id) as total_banks,
    COUNT(r.review_id) as total_reviews,
    ROUND(AVG(r.rating), 2) as overall_avg_rating,
    COUNT(DISTINCT r.review_date) as unique_dates,
    MIN(r.review_date) as earliest_review,
    MAX(r.review_date) as latest_review
FROM banks b
LEFT JOIN reviews r ON b.bank_id = r.bank_id;

