-- PostgreSQL Database Setup Script


-- Create Banks table
CREATE TABLE IF NOT EXISTS banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL UNIQUE,
    app_name VARCHAR(200) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Reviews table
CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY,
    bank_id INTEGER NOT NULL,
    review_text TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review_date DATE,
    sentiment_label VARCHAR(20),
    sentiment_score DECIMAL(5, 4),
    source VARCHAR(50) DEFAULT 'Google Play Store',
    themes TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment_label);
CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(review_date);

-- Insert bank data
INSERT INTO banks (bank_name, app_name) VALUES
    ('CBE', 'Commercial Bank of Ethiopia Mobile'),
    ('BOA', 'Bank of Abyssinia Mobile'),
    ('Dashen', 'Dashen Bank Mobile')
ON CONFLICT (bank_name) DO NOTHING;

-- Verification queries (commented out - run separately)
-- Count reviews per bank
-- SELECT b.bank_name, COUNT(r.review_id) as review_count
-- FROM banks b
-- LEFT JOIN reviews r ON b.bank_id = r.bank_id
-- GROUP BY b.bank_name
-- ORDER BY review_count DESC;

-- Average rating per bank
-- SELECT b.bank_name, 
--        ROUND(AVG(r.rating), 2) as avg_rating,
--        COUNT(r.review_id) as review_count
-- FROM banks b
-- LEFT JOIN reviews r ON b.bank_id = r.bank_id
-- GROUP BY b.bank_name
-- ORDER BY avg_rating DESC;

-- Sentiment distribution
-- SELECT b.bank_name, 
--        r.sentiment_label,
--        COUNT(*) as count,
--        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY b.bank_name), 2) as percentage
-- FROM banks b
-- JOIN reviews r ON b.bank_id = r.bank_id
-- GROUP BY b.bank_name, r.sentiment_label
-- ORDER BY b.bank_name, r.sentiment_label;

