import sqlite3
import random
import uuid
import pandas as pd
import re
from faker import Faker
import nltk

# ==================== CONFIGURATION ====================
DB_PATH = "simulated_data/fake_reviews.db"
TABLE_NAME = "fake_reviews" 
CSV_PATH = "simulated_data/fake_reviews.csv"
NUM_REVIEWS_TO_GENERATE = 1357

# Initialize Faker to generate fake names
fake = Faker()

# --- NEW: Define Stopwords (to match your dashboard) ---
try:
    stopwords_en = set(nltk.corpus.stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stopwords_en = set(nltk.corpus.stopwords.words("english"))

CUSTOM_STOPWORDS = {
    'laundry', 'clothes', 'cloth', 'wash', 'selava',
    'service', 'services', 'company', 'local', 'place',
    'time', 'times', 'do', 'made', 'done'
}
ALL_STOPWORDS = stopwords_en.union(CUSTOM_STOPWORDS)


# ==================== COMMENT TEMPLATES ====================
POSITIVE_COMMENTS = [
    "Excellent service! My clothes came back perfect.",
    "Very fast and professional. Highly recommended!",
    "The quality is amazing and the staff is very friendly.",
    "I'm a regular customer and I'm always satisfied. Great job!",
    "Fantastic results. My delicate clothes were handled with care.",
    "Super convenient and reliable. Wouldn't go anywhere else.",
    "The best laundry service in town, without a doubt."
]
NEUTRAL_COMMENTS = [
    "The service was okay, did the job.",
    "It's decent, but a bit expensive for what it is.",
    "Took a little longer than I expected, but the result was fine.",
    "The quality is average. Nothing special to report.",
    "It works.",
    "A standard laundry service. It meets basic expectations."
]
NEGATIVE_COMMENTS = [
    "Terrible experience. They lost one of my shirts and denied it.",
    "Took way too long to get my clothes back. Very slow service.",
    "My clothes came back with a strange smell. Very disappointed.",
    "Poor customer service. The attendant was rude.",
    "Overpriced and the quality is subpar. I will not be returning.",
    "They damaged my favorite jacket. Unacceptable!",
    "The clothes were still slightly damp when I picked them up."
]

# ==================== HELPER FUNCTION (UPDATED) ====================

# --- UPDATED with your new cleaning logic ---
def process_clean_comment(text, stopwords_set):
    """
    Applies the specific text cleaning and stopword removal process.
    """
    # Make sure text is a string and convert to lowercase
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove characters that are not letters or spaces
    # Note: I've adapted your regex for English characters.
    text = re.sub(r"[^a-z\s]", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stopwords_set])
    return text

# ==================== MAIN FUNCTIONS ====================
def generate_fake_reviews(num_reviews):
    """Generates a list of fake reviews."""
    reviews = []
    print(f"Generating {num_reviews} fake reviews...")

    for _ in range(num_reviews):
        rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.05, 0.20, 0.35, 0.35], k=1)[0]
        
        if rating >= 4:
            comment = random.choice(POSITIVE_COMMENTS)
        elif rating == 3:
            comment = random.choice(NEUTRAL_COMMENTS)
        else:
            comment = random.choice(NEGATIVE_COMMENTS)
            
        review_id = str(uuid.uuid4())
        random_days = random.randint(1, 730)
        
        # 1. Generate a timezone-aware timestamp in UTC
        date_object = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=random_days, hours=random.randint(0,23))
        
        # --- FIX IS HERE ---
        # 2. Format it into the EXACT string format of your real data
        date_string = date_object.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
        
        user = fake.name()
        clean_comment = process_clean_comment(comment, ALL_STOPWORDS)
        
        # 3. Append the correctly formatted string to the list
        reviews.append((review_id, date_string, user, rating, comment, clean_comment))
        
    print("Generation complete.")
    return reviews

def insert_reviews_into_db(db_path, table_name, reviews_data):
    """Connects to the DB and inserts the review data."""
    print(f"Connecting to database at '{db_path}'...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        review_id TEXT PRIMARY KEY,
        date TIMESTAMP,
        user TEXT,
        rating INTEGER,
        comment TEXT,
        clean_comment TEXT
    )
    """)
    
    insert_query = f"INSERT INTO {table_name} (review_id, date, user, rating, comment, clean_comment) VALUES (?, ?, ?, ?, ?, ?)"
    cursor.executemany(insert_query, reviews_data)
    
    conn.commit()
    conn.close()
    
    print(f"Successfully inserted {len(reviews_data)} new reviews into the database!")

def save_reviews_to_csv(csv_path, reviews_data):
    """Saves the generated reviews to a CSV file."""
    print(f"Saving data to CSV file at '{csv_path}'...")
    columns = ['review_id', 'date', 'user', 'rating', 'comment', 'clean_comment']
    df = pd.DataFrame(reviews_data, columns=columns)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Successfully saved {len(reviews_data)} reviews to {csv_path}!")

# ==================== SCRIPT EXECUTION ====================
if __name__ == "__main__":
    generated_reviews = generate_fake_reviews(NUM_REVIEWS_TO_GENERATE)
    save_reviews_to_csv(CSV_PATH, generated_reviews)
    insert_reviews_into_db(DB_PATH, TABLE_NAME, generated_reviews)
    