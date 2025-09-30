import pandas as pd
import sqlite3

df = pd.read_csv("data/clean_reviews.csv")
conn = sqlite3.connect("db/reviews.db")

# create table
conn.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    date TIMESTAMP,
    user TEXT,
    rating INTEGER,
    comment TEXT,
    clean_comment TEXT
)
""")

#insert/replace data from csv to table
df.to_sql("reviews", conn, if_exists="replace", index=False)
conn.close()