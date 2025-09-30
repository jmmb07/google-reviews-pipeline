import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

import nltk
try:
    stopwords.words("portuguese")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("portuguese")) #words that are not useful

df = pd.read_csv("data/reviews.csv")

# clean comments
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text) #remove urls
    text = re.sub(r"[^a-záéíóúãõâêîôûç\s]", " ", text) #remove caracthers that are not words or spaces
    text = re.sub(r"\s+", " ", text).strip() #remove double or long spaces
    return text

df["clean_comment"] = df["comment"].apply(clean_text)
df["clean_comment"] = df["clean_comment"].apply(
    lambda x: " ".join([w for w in x.split() if w not in STOPWORDS]) #ignore stopwords
)

# convert rating and date
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df.to_csv("data/clean_reviews.csv", index=False)
print("Transform completed, clean CSV saved in data/clean_reviews.csv")
