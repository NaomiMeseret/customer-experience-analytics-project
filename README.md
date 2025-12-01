# 📊 Customer Experience Analytics for Fintech Apps

A Real-World Data Engineering Challenge: Scraping, Analyzing, and Visualizing Google Play Store Reviews

## 📋 Project Overview

This project analyzes customer satisfaction with mobile banking apps by collecting and processing user reviews from the Google Play Store for three Ethiopian banks:

- 🏦 **Commercial Bank of Ethiopia (CBE)**
- 🏦 **Bank of Abyssinia (BOA)**
- 🏦 **Dashen Bank**

## 🎯 Business Objective

Omega Consultancy is supporting banks to improve their mobile apps to enhance customer retention and satisfaction. This project:

- 🕷️ Scrapes user reviews from the Google Play Store
- 💭 Analyzes sentiment (positive/negative/neutral) and extracts themes
- 🔍 Identifies satisfaction drivers and pain points
- 💾 Stores cleaned review data for analysis
- 📈 Delivers insights with visualizations and actionable recommendations

## 📁 Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── run_task1.py              # Helper script to run Task 1
├── run_task2.py              # Helper script to run Task 2
├── task1_data_collection/
│   ├── scrape_reviews.py     # Web scraping script
│   └── preprocess_reviews.py # Data cleaning script
├── task2_analysis/
│   ├── sentiment_analysis.py # Sentiment analysis using DistilBERT
│   └── thematic_analysis.py  # Theme extraction and clustering
└── data/
    ├── raw/                  # Raw scraped data
    └── processed/            # Cleaned and analyzed data
```

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/NaomiMeseret/customer-experience-analytics-project.git
cd customer-experience-analytics-project
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4️⃣ Update App IDs (⚠️ IMPORTANT)

Before running the scraping script, you need to update the app IDs in `task1_data_collection/scrape_reviews.py` with the actual Google Play Store app IDs for the three banks:

1. Find the app IDs by searching for each bank's app on Google Play Store
2. The app ID is typically in the URL: `https://play.google.com/store/apps/details?id=APP_ID_HERE`
3. Update the `BANK_APPS` dictionary in `scrape_reviews.py`

Example:

```python
BANK_APPS = {
    'CBE': {
        'app_id': 'com.cbe.mobilebanking',  # Replace with actual ID
        'app_name': 'Commercial Bank of Ethiopia Mobile'
    },
    # ... etc
}
```

### 5️⃣ Run Task 1: Data Collection

```bash
# Switch to task-1 branch
git checkout -b task-1

# Option 1: Run scripts individually
python task1_data_collection/scrape_reviews.py
python task1_data_collection/preprocess_reviews.py

# Option 2: Use helper script (runs both in sequence)
python run_task1.py
```

### 6️⃣ Run Task 2: Sentiment and Thematic Analysis

```bash
# Switch to task-2 branch
git checkout -b task-2

# Option 1: Run scripts individually
python task2_analysis/sentiment_analysis.py
python task2_analysis/thematic_analysis.py

# Option 2: Use helper script (runs both in sequence)
python run_task2.py
```

> **💡 Note**: The first run will download the DistilBERT model (~250MB), which may take a few minutes.

## 📥 Task 1: Data Collection and Preprocessing

### 🔬 Methodology

1. **🕷️ Web Scraping**: Uses `google-play-scraper` library to collect reviews from Google Play Store

   - Scrapes reviews sorted by newest first
   - Collects review text, rating (1-5 stars), date, and metadata
   - Targets 400+ reviews per bank (1,200+ total)
   - Includes rate limiting to be respectful with API calls

2. **🧹 Preprocessing**:

   - Remove duplicates based on review text and bank
   - Handle missing data (fill missing ratings with bank median, handle missing dates)
   - Normalize dates to YYYY-MM-DD format
   - Clean review text (remove extra whitespace)
   - Validate ratings (ensure 1-5 range)
   - Add unique review_id for tracking

3. **📊 Data Schema**:
   - `review_id`: Unique identifier
   - `review`: Review text
   - `rating`: 1-5 star rating
   - `date`: Review date (YYYY-MM-DD format)
   - `bank`: Bank name (CBE, BOA, Dashen)
   - `source`: Data source (Google Play Store)

### 📤 Output

- 📄 Raw data: `data/raw/all_reviews_raw.csv` and individual bank files
- ✨ Clean CSV dataset: `data/processed/reviews_cleaned.csv`

## 🧠 Task 2: Sentiment and Thematic Analysis

### 🔬 Methodology

1. **💭 Sentiment Analysis**:

   - Uses `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face
   - Computes sentiment scores (positive, negative, neutral)
   - Aggregates by bank and rating

2. **🏷️ Thematic Analysis**:
   - Keyword extraction using TF-IDF (Term Frequency-Inverse Document Frequency)
   - Additional keyword extraction using spaCy NLP (if available)
   - Theme identification through keyword matching
   - Groups keywords into 3-7 themes per bank:
     - 🔐 **Account Access Issues**: Login, password, authentication problems
     - ⚡ **Transaction Performance**: Transfer speed, processing time, delays
     - 🎨 **User Interface & Experience**: UI design, navigation, usability
     - 🎧 **Customer Support**: Support quality, response time, assistance
     - ✨ **Feature Requests**: Missing features, enhancement requests
     - 🐛 **App Reliability**: Crashes, bugs, stability issues
     - 💳 **Payment & Banking Features**: Payment methods, banking functionality

### 📤 Output

- 💭 Sentiment analysis results in `data/processed/reviews_with_sentiment.csv`
- 🏷️ Thematic analysis results in `data/processed/reviews_with_themes.csv`

## 📊 Key Performance Indicators (KPIs)

### 📥 Task 1

- ✅ 1,200+ reviews collected with <5% missing data
- ✅ Clean CSV dataset
- ✅ Organized Git repo with clear commits

### 🧠 Task 2

- ✅ Sentiment scores for 90%+ reviews
- ✅ 3+ themes per bank with examples
- ✅ Modular pipeline code

## 🗄️ Task 3: Database Storage (PostgreSQL)

### 🔬 Methodology

1. **Database Setup**:

   - PostgreSQL database named `bank_reviews`
   - Two main tables: `banks` and `reviews`
   - Foreign key relationships for data integrity
   - Indexes for query performance

2. **Database Schema**:

   **Banks Table**:

   - `bank_id` (SERIAL PRIMARY KEY): Unique identifier
   - `bank_name` (VARCHAR): Bank name (CBE, BOA, Dashen)
   - `app_name` (VARCHAR): Full app name
   - `created_at` (TIMESTAMP): Record creation timestamp

   **Reviews Table**:

   - `review_id` (INTEGER PRIMARY KEY): Unique review identifier
   - `bank_id` (INTEGER FOREIGN KEY): References banks table
   - `review_text` (TEXT): Review content
   - `rating` (INTEGER): 1-5 star rating
   - `review_date` (DATE): Review posting date
   - `sentiment_label` (VARCHAR): POSITIVE/NEGATIVE
   - `sentiment_score` (DECIMAL): Sentiment confidence score
   - `source` (VARCHAR): Data source (Google Play Store)
   - `themes` (TEXT[]): Array of identified themes
   - `created_at` (TIMESTAMP): Record creation timestamp

3. **Data Insertion**:
   - Python script using `psycopg2` for database operations
   - Batch insertion for efficiency
   - Data validation and error handling

### 📤 Output

- ✅ PostgreSQL database with schema
- ✅ 1,000+ reviews stored in database
- ✅ SQL verification queries
- ✅ Schema documentation

## 📈 Task 4: Insights and Recommendations

### 🔬 Methodology

1. **Insights Generation**:

   - Identify 2+ satisfaction drivers per bank (e.g., fast navigation, good UI)
   - Identify 2+ pain points per bank (e.g., crashes, slow loading)
   - Compare banks across key metrics (ratings, sentiment, themes)

2. **Visualizations** (3-5 plots):

   - Rating distribution by bank
   - Sentiment trends and analysis
   - Theme distribution analysis
   - Comparative dashboard
   - Keyword cloud visualization

3. **Recommendations**:

   - 2+ actionable improvements per bank
   - Prioritized by impact and feasibility
   - Evidence-based with review examples

4. **Ethical Considerations**:
   - Review bias documentation
   - Limitations and mitigation strategies

### 📤 Output

- 📊 5 key visualizations
- 📝 Insights and recommendations report
- 🔍 Bank comparison analysis
- ⚠️ Ethical considerations documentation

### 📥 Task 3 KPIs

- ✅ Working database connection + insert script
- ✅ Tables populated with >1,000 review entries
- ✅ SQL dump or schema file committed to GitHub

### 📈 Task 4 KPIs

- ✅ 2+ drivers/pain points with evidence per bank
- ✅ Clear, labeled visualizations (3-5 plots)
- ✅ Practical recommendations per bank

## 🛠️ Technologies Used

- 🕷️ **Web Scraping**: google-play-scraper
- 🤖 **NLP**: Transformers (DistilBERT), spaCy, scikit-learn
- 📊 **Data Processing**: Pandas, NumPy
- 🧠 **Machine Learning**: Hugging Face Transformers, PyTorch
- 🗄️ **Database**: PostgreSQL, psycopg2
- 📈 **Visualization**: Matplotlib, Seaborn, WordCloud
- 🔀 **Version Control**: Git, GitHub

## ⚠️ Important Notes

1. **🔑 App IDs**: You must update the app IDs in `scrape_reviews.py` with the actual Google Play Store app IDs before running the scraper.

2. **📥 Model Download**: The DistilBERT model will be automatically downloaded on first run (~250MB). Ensure you have a stable internet connection.

3. **🔒 Data Privacy**: This project is for educational purposes. Ensure compliance with Google Play Store's terms of service when scraping reviews.

4. **⏱️ Rate Limiting**: The scraper includes delays between requests to be respectful. Scraping 1,200+ reviews may take 10-20 minutes.

5. **📦 Dependencies**: Make sure to install spaCy's English model: `python -m spacy download en_core_web_sm`

6. **🗄️ PostgreSQL**: Install PostgreSQL and create the `bank_reviews` database before running Task 3 scripts.
