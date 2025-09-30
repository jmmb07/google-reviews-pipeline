import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
import altair as alt
from collections import Counter
import nltk
from nltk.util import ngrams
import re

# Download NLTK data (only if necessary)
try:
    stopwords_en = set(nltk.corpus.stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stopwords_en = set(nltk.corpus.stopwords.words("english"))

# ==================== SETTINGS ====================
st.set_page_config(
    page_title="Review Dashboard - Simulated Data",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = Path("db/reviews.db")
TABLE_NAME = "reviews"
# DB_PATH = "simulated_data/fake_reviews.db"
# TABLE_NAME = "fake_reviews" 

# CUSTOM STOPWORDS
CUSTOM_STOPWORDS = {
    'laundry', 'clothes', 'cloth', 'wash',
    'service', 'services', 'company', 'local', 'place',
    'time', 'times', 'do', 'made', 'done'
}
ALL_STOPWORDS = stopwords_en.union(CUSTOM_STOPWORDS)

# ==================== DATA FUNCTIONS ====================
@st.cache_data
def load_data_from_sqlite(db_path: str, table_name: str = "reviews") -> pd.DataFrame:
    """Loads data from SQLite with error handling."""
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(str(db_file))
    try:
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, conn, parse_dates=["date"])
    finally:
        conn.close()
    
    # Normalization
    df.columns = [c.lower() for c in df.columns]
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_convert(None)
    
    return df

# ==================== NLP FUNCTIONS ====================
def clean_text_advanced(text):
    """Advanced text cleaning."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

def extract_top_words(texts, n_top=10, min_word_length=3):
    """Extracts top words after removing stopwords."""
    words = []
    for text in texts:
        if pd.isna(text):
            continue
        for word in str(text).split():
            word = word.lower()
            if word not in ALL_STOPWORDS and len(word) >= min_word_length:
                words.append(word)
    
    counter = Counter(words)
    return counter.most_common(n_top)

def extract_ngrams(texts, n=2, top_n=10):
    """Extracts the most common bigrams or trigrams."""
    all_ngrams = []
    
    for text in texts:
        if pd.isna(text):
            continue
        words = [w.lower() for w in str(text).split() 
                if w.lower() not in ALL_STOPWORDS and len(w) > 2]
        
        text_ngrams = list(ngrams(words, n))
        all_ngrams.extend([' '.join(ng) for ng in text_ngrams])
    
    counter = Counter(all_ngrams)
    return counter.most_common(top_n)

def classify_sentiment(rating):
    """Classifies sentiment based on rating."""
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

def classify_sentiment_advanced(rating, comment=None):
    """
    Hybrid classification: rating + text analysis.
    """
    # Base classification by rating
    base_sentiment = classify_sentiment(rating)
    
    # If there's no comment, return the base classification
    if not comment or pd.isna(comment) or base_sentiment == "Undefined":
        return base_sentiment
    
    # Simple text analysis (keywords)
    comment_lower = str(comment).lower()
    
    positive_words = ['excellent', 'great', 'wonderful', 'perfect', 'loved', 'recommend']
    negative_words = ['terrible', 'horrible', 'awful', 'never again', 'disappointing', 'bad']
    
    positive_count = sum(1 for word in positive_words if word in comment_lower)
    negative_count = sum(1 for word in negative_words if word in comment_lower)
    
    # Adjust sentiment if there's a strong contradiction
    if base_sentiment == "Positive" and negative_count > positive_count + 2:
        return "Neutral"  # High rating but very negative comment
    
    if base_sentiment == "Negative" and positive_count > negative_count + 2:
        return "Neutral"  # Low rating but positive comment
    
    return base_sentiment

@st.cache_data
def generate_wordcloud(text, sentiment):
    """Generates a word cloud and returns it as bytes (with caching)."""
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from io import BytesIO
    
    if not text.strip():
        return None
    
    # Generate word cloud
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=ALL_STOPWORDS,
        collocations=False,
        colormap="viridis",
        max_words=50,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="PNG", bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    
    return buf.getvalue()

# ==================== LOAD DATA ====================
st.title("🧼 Review Dashboard - Simulated Data")

try:
    df = load_data_from_sqlite(str(DB_PATH), TABLE_NAME)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ==================== SIDEBAR: FILTERS ====================
st.sidebar.header("🔍 Filters")

# Date filter
if "date" in df.columns:
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.sidebar.date_input(
        "📅 Date range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        st.warning("Please select a start and end date.")
        st.stop()
else:
    start_date, end_date = None, None

# Rating filter
if "rating" in df.columns:
    min_rating, max_rating = st.sidebar.slider(
        "⭐ Rating range",
        min_value=int(df["rating"].min()),
        max_value=int(df["rating"].max()),
        value=(int(df["rating"].min()), int(df["rating"].max())),
        step=1
    )
else:
    min_rating, max_rating = None, None

# Editable custom stopwords
st.sidebar.divider()
st.sidebar.subheader("🚫 Words to Ignore")
st.sidebar.caption("Add words that don't add value to the analysis (e.g., company name)")

# Text input to add words
user_stopwords_input = st.sidebar.text_area(
    "Enter words separated by a comma:",
    value="laundry, clothes, wash, service, services",
    height=100,
    help="Common context-specific words to be ignored in the analysis"
)

# Process user's stopwords
user_stopwords = set()
if user_stopwords_input:
    user_stopwords = {w.strip().lower() for w in user_stopwords_input.split(',') if w.strip()}

# Combine all stopwords
ALL_STOPWORDS = stopwords_en.union(CUSTOM_STOPWORDS).union(user_stopwords)

st.sidebar.info(f"📊 Total words ignored: {len(ALL_STOPWORDS)}")

# Apply filters
df_filtered = df.copy()
if start_date and end_date:
    df_filtered = df_filtered[(df_filtered["date"] >= start_date) & (df_filtered["date"] <= end_date)]
if min_rating is not None and max_rating is not None:
    df_filtered = df_filtered[(df_filtered["rating"] >= min_rating) & (df_filtered["rating"] <= max_rating)]

# Add sentiment column
if "rating" in df_filtered.columns:
    df_filtered["sentiment"] = df_filtered["rating"].apply(classify_sentiment_advanced)

st.sidebar.success(f"✅ {len(df_filtered)} reviews filtered")

# ==================== MAIN TABS ====================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💬 Text Analysis", "📈 Trends", "📋 Quality"])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    st.header("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", len(df_filtered))
    with col2:
        avg_rating = round(df_filtered["rating"].mean(), 2) if "rating" in df_filtered.columns else 0
        st.metric("Average Rating", f"⭐ {avg_rating}")
    with col3:
        unique_users = df_filtered["user"].nunique() if "user" in df_filtered.columns else 0
        st.metric("Unique Users", unique_users)
    with col4:
        if "sentiment" in df_filtered.columns:
            positive_pct = (df_filtered["sentiment"] == "Positive").sum() / len(df_filtered) * 100
            st.metric("% Positive", f"{positive_pct:.1f}%")
    
    st.divider()
    
    # Side-by-side charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Rating Distribution")
        if "rating" in df_filtered.columns:
            rating_counts = df_filtered["rating"].value_counts().sort_index().reset_index()
            rating_counts.columns = ["rating", "count"]
            
            bar_chart = alt.Chart(rating_counts).mark_bar(color='steelblue').encode(
                x=alt.X("rating:O", title="Rating"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["rating", "count"]
            ).properties(height=300)
            
            st.altair_chart(bar_chart, use_container_width=True)
    
    with col_right:
        st.subheader("Sentiment Distribution")
        if "sentiment" in df_filtered.columns:
            sentiment_counts = df_filtered["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["sentiment", "count"]
            
            pie_chart = alt.Chart(sentiment_counts).mark_arc().encode(
                theta=alt.Theta(field="count", type="quantitative"),
                color=alt.Color(
                    field="sentiment",
                    type="nominal",
                    scale=alt.Scale(
                        domain=["Positive", "Neutral", "Negative"],
                        range=["#2ecc71", "#f39c12", "#e74c3c"]
                    )
                ),
                tooltip=["sentiment", "count"]
            ).properties(height=300)
            
            st.altair_chart(pie_chart, use_container_width=True)
    
    st.divider()
    
    # Time trend
    st.subheader("Average Rating Over Time")
    if "date" in df_filtered.columns and "rating" in df_filtered.columns:
        df_filtered["month"] = df_filtered["date"].dt.to_period("M").dt.to_timestamp()
        rating_trend = df_filtered.groupby("month")["rating"].mean().reset_index()
        
        line_chart = alt.Chart(rating_trend).mark_line(point=True, color='steelblue').encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("rating:Q", title="Average Rating", scale=alt.Scale(domain=[0, 5])),
            tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("rating:Q", format=".2f")]
        ).properties(height=300)
        
        st.altair_chart(line_chart, use_container_width=True)

# ==================== TAB 2: TEXT ANALYSIS ====================
with tab2:
    st.header("💬 Text Analysis and NLP")
    
    analysis_type = st.radio(
        "Choose the type of analysis:",
        ["Single Words", "Phrases (Bigrams)", "Word Cloud"],
        horizontal=True,
        key="analysis_type_radio"
    )
    
    st.divider()
    
    for sentiment in ["Positive", "Neutral", "Negative"]:
        sentiment_data = df_filtered[df_filtered["sentiment"] == sentiment]
        
        if len(sentiment_data) == 0:
            continue
        
        # Choose color and emoji by sentiment
        emoji = "😊" if sentiment == "Positive" else "😐" if sentiment == "Neutral" else "😞"
        
        with st.expander(f"{emoji} {sentiment} ({len(sentiment_data)} reviews)", expanded=True):
            
            # OPTION 1: Single Words (word count)
            if analysis_type == "Single Words":
                top_words = extract_top_words(sentiment_data["clean_comment"], n_top=15)
                
                if top_words:
                    df_words = pd.DataFrame(top_words, columns=["Word", "Frequency"])
                    color = "#2ecc71" if sentiment == "Positive" else "#f39c12" if sentiment == "Neutral" else "#e74c3c"

                    chart = alt.Chart(df_words).mark_bar().encode(
                        x=alt.X("Frequency:Q"),
                        y=alt.Y("Word:N", sort="-x"),
                        color=alt.value(color),
                        tooltip=["Word", "Frequency"]
                    ).properties(height=400)
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Not enough words for analysis.")
            
            # OPTION 2: Phrases (Bigrams)
            elif analysis_type == "Phrases (Bigrams)":
                bigrams = extract_ngrams(sentiment_data["clean_comment"], n=2, top_n=10)
                
                if bigrams:
                    df_bigrams = pd.DataFrame(bigrams, columns=["Phrase", "Frequency"])
                    color = "#2ecc71" if sentiment == "Positive" else "#f39c12" if sentiment == "Neutral" else "#e74c3c"
                    
                    chart = alt.Chart(df_bigrams).mark_bar().encode(
                        x=alt.X("Frequency:Q"),
                        y=alt.Y("Phrase:N", sort="-x"),
                        color=alt.value(color),
                        tooltip=["Phrase", "Frequency"]
                    ).properties(height=350)
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Not enough data to generate bigrams.")
            
            # OPTION 3: Word Cloud
            elif analysis_type == "Word Cloud":
                text = " ".join([str(c) for c in sentiment_data["clean_comment"].dropna() if c])
                
                wordcloud_image = generate_wordcloud(text, sentiment)
                
                if wordcloud_image:
                    st.image(wordcloud_image, use_container_width=False, width=800)
                else:
                    st.info("Not enough text to generate a word cloud.")

# ==================== TAB 3: TRENDS ====================
with tab3:
    st.header("📈 Temporal Analysis and Patterns")
    
    if "date" in df_filtered.columns:
        df_filtered["day_of_week"] = df_filtered["date"].dt.day_name()
        df_filtered["hour"] = df_filtered["date"].dt.hour
        df_filtered["is_weekend"] = df_filtered["date"].dt.dayofweek >= 5
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Reviews by Day of the Week")
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_counts = df_filtered["day_of_week"].value_counts().reindex(dow_order, fill_value=0).reset_index()
            dow_counts.columns = ["day", "count"]
            
            chart = alt.Chart(dow_counts).mark_bar(color='steelblue').encode(
                x=alt.X("day:N", title="Day of the Week", sort=dow_order),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["day", "count"]
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
        
        with col2:
            st.subheader("Average Rating: Weekday vs. Weekend")
            weekend_avg = df_filtered.groupby("is_weekend")["rating"].mean().reset_index()
            weekend_avg["period"] = weekend_avg["is_weekend"].map({True: "Weekend", False: "Weekday"})
            
            chart = alt.Chart(weekend_avg).mark_bar(color='coral').encode(
                x=alt.X("period:N", title=""),
                y=alt.Y("rating:Q", title="Average Rating", scale=alt.Scale(domain=[0, 5])),
                tooltip=["period", alt.Tooltip("rating:Q", format=".2f")]
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
        
        st.divider()
        
        # HEATMAP
        st.subheader("📊 Heatmap: Average Rating by Day and Hour")
        
        # Prepare data
        heatmap_data = df_filtered.groupby(["day_of_week", "hour"])["rating"].mean().reset_index()
        
        # Create heatmap with Altair
        heatmap = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X("hour:O", title="Hour of Day"),
            y=alt.Y("day_of_week:N", title="Day of the Week", sort=dow_order),
            color=alt.Color("rating:Q", scale=alt.Scale(scheme="goldred"), title="Average Rating"),
            tooltip=[
                alt.Tooltip("day_of_week:N", title="Day"),
                alt.Tooltip("hour:O", title="Hour"),
                alt.Tooltip("rating:Q", format=".2f", title="Rating")
            ]
        ).properties(width=700, height=400)
        
        st.altair_chart(heatmap, use_container_width=True)

# ==================== TAB 4: DATA QUALITY ====================
with tab4:
    st.header("📋 Data Quality and Integrity")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        duplicates = df.duplicated().sum()
        st.metric("Duplicates", duplicates)
    
    with col2:
        empty_comments = df["comment"].isna().sum()
        st.metric("Empty Comments", empty_comments)
    
    with col3:
        data_freshness = (pd.Timestamp.now() - df["date"].max()).days
        st.metric("Last Review", f"{data_freshness} days ago")
    
    st.divider()
    
    st.subheader("Completeness by Column")
    completeness = (1 - df.isnull().sum() / len(df)) * 100
    df_completeness = completeness.reset_index()
    df_completeness.columns = ["Column", "Completeness (%)"]
    df_completeness["Completeness (%)"] = df_completeness["Completeness (%)"].round(2)
    
    chart = alt.Chart(df_completeness).mark_bar(color='teal').encode(
        x=alt.X("Completeness (%):Q", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Column:N", sort="-x"),
        tooltip=["Column", "Completeness (%)"]
    ).properties(height=200)
    
    st.altair_chart(chart, use_container_width=True)
    
    st.divider()
    
    # Data Download
    st.subheader("💾 Export Data")
    
    col_download1, col_download2, col_download3 = st.columns(3)
    
    with col_download1:
        # CSV with UTF-8 BOM (opens correctly in Excel)
        csv_utf8_bom = df_filtered.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📊 CSV (UTF-8)",
            data=csv_utf8_bom,
            file_name=f"filtered_reviews_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help=".csv"
        )
    
    with col_download2:
        # Excel (XLSX) - native format
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_filtered.to_excel(writer, sheet_name='Reviews', index=False)
        
        st.download_button(
            label="📗 Excel (XLSX)",
            data=buffer.getvalue(),
            file_name=f"filtered_reviews_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.ms-excel",
            help=".xlsx"
        )
    
    with col_download3:
        # JSON (for developers)
        json_data = df_filtered.to_json(orient='records', force_ascii=False, indent=2)
        st.download_button(
            label="🔧 JSON",
            data=json_data,
            file_name=f"filtered_reviews_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            help=".json"
        )