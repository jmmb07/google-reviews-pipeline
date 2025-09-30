import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
import altair as alt
from collections import Counter
import nltk
from nltk.util import ngrams
import re

# Download NLTK data (apenas se necessário)
try:
    stopwords_pt = set(nltk.corpus.stopwords.words("portuguese"))
except LookupError:
    nltk.download("stopwords")
    stopwords_pt = set(nltk.corpus.stopwords.words("portuguese"))

# ==================== CONFIGURAÇÕES ====================
st.set_page_config(
    page_title="Dashboard de Reviews - SELAVA",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = Path("db/reviews.db")
TABLE_NAME = "reviews"

# STOPWORDS PERSONALIZADAS
CUSTOM_STOPWORDS = {
    'lavanderia', 'roupas', 'roupa', 'lavar', 'selava',
    'serviço', 'serviços', 'empresa', 'local', 'lugar',
    'vez', 'vezes', 'fazer', 'feito', 'feita'
}
ALL_STOPWORDS = stopwords_pt.union(CUSTOM_STOPWORDS)

# ==================== FUNÇÕES DE DADOS ====================
@st.cache_data
def load_data_from_sqlite(db_path: str, table_name: str = "reviews") -> pd.DataFrame:
    """Carrega dados do SQLite com tratamento de erros."""
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"Banco de dados não encontrado: {db_path}")
    
    conn = sqlite3.connect(str(db_file))
    try:
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, conn, parse_dates=["date"])
    finally:
        conn.close()
    
    # Normalização
    df.columns = [c.lower() for c in df.columns]
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_convert(None)
    
    return df

# ==================== FUNÇÕES DE NLP ====================
def clean_text_advanced(text):
    """Limpeza avançada de texto."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # remove pontuação
    text = re.sub(r'\s+', ' ', text).strip()  # remove espaços extras
    return text

def extract_top_words(texts, n_top=10, min_word_length=3):
    """Extrai top palavras removendo stopwords."""
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
    """Extrai bigramas ou trigramas mais comuns """
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
    """Classifica sentimento baseado em rating."""
    if rating >= 4:
        return "Positivo"
    elif rating == 3:
        return "Neutro"
    else:
        return "Negativo"

def classify_sentiment_advanced(rating, comment=None):
    """
    Classificação híbrida: rating + análise de texto.
    """
    # Classificação base por rating
    base_sentiment = classify_sentiment(rating)
    
    # Se não tem comentário, retorna classificação base
    if not comment or pd.isna(comment) or base_sentiment == "Indefinido":
        return base_sentiment
    
    # Análise simples do texto (palavras-chave)
    comment_lower = str(comment).lower()
    
    positive_words = ['excelente', 'ótimo', 'maravilhoso', 'perfeito', 'adorei', 'recomendo']
    negative_words = ['péssimo', 'horrível', 'terrível', 'nunca mais', 'decepcionante', 'ruim', 'péssima']
    
    positive_count = sum(1 for word in positive_words if word in comment_lower)
    negative_count = sum(1 for word in negative_words if word in comment_lower)
    
    # 4. Ajustar sentimento se houver contradição forte
    if base_sentiment == "Positivo" and negative_count > positive_count + 2:
        return "Neutro"  # Rating alto mas comentário muito negativo
    
    if base_sentiment == "Negativo" and positive_count > negative_count + 2:
        return "Neutro"  # Rating baixo mas comentário positivo
    
    return base_sentiment

@st.cache_data
def generate_wordcloud(text, sentiment):
    """Gera word cloud e retorna como bytes (com cache)."""
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from io import BytesIO
    
    if not text.strip():
        return None
    
    # Gerar word cloud
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
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Salvar em buffer
    buf = BytesIO()
    plt.savefig(buf, format="PNG", bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    
    return buf.getvalue()

# ==================== CARREGAR DADOS ====================
st.title("🧼 Dashboard de Avaliações - SELAVA")

try:
    df = load_data_from_sqlite(str(DB_PATH), TABLE_NAME)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()

# ==================== SIDEBAR: FILTROS ====================
st.sidebar.header("🔍 Filtros")

# Filtro de data
if "date" in df.columns:
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.sidebar.date_input(
        "📅 Intervalo de datas",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        st.warning("Selecione data inicial e final.")
        st.stop()
else:
    start_date, end_date = None, None

# Filtro de rating
if "rating" in df.columns:
    min_rating, max_rating = st.sidebar.slider(
        "⭐ Intervalo de notas",
        min_value=int(df["rating"].min()),
        max_value=int(df["rating"].max()),
        value=(int(df["rating"].min()), int(df["rating"].max())),
        step=1
    )
else:
    min_rating, max_rating = None, None

# Stopwords personalizadas editáveis
st.sidebar.divider()
st.sidebar.subheader("🚫 Palavras a Ignorar")
st.sidebar.caption("Adicione palavras que não agregam à análise (ex: nome da empresa)")

# Input de texto para adicionar palavras
user_stopwords_input = st.sidebar.text_area(
    "Digite palavras separadas por vírgula:",
    value="lavanderia, roupas, roupa, lavar, selava, serviço, serviços",
    height=100,
    help="Palavras comuns ao contexto que devem ser ignoradas na análise"
)

# Processar stopwords do usuário
user_stopwords = set()
if user_stopwords_input:
    user_stopwords = {w.strip().lower() for w in user_stopwords_input.split(',') if w.strip()}

# Combinar todas as stopwords
ALL_STOPWORDS = stopwords_pt.union(CUSTOM_STOPWORDS).union(user_stopwords)

st.sidebar.info(f"📊 Total de palavras ignoradas: {len(ALL_STOPWORDS)}")

# Aplicar filtros
df_filtered = df.copy()
if start_date and end_date:
    df_filtered = df_filtered[(df_filtered["date"] >= start_date) & (df_filtered["date"] <= end_date)]
if min_rating is not None and max_rating is not None:
    df_filtered = df_filtered[(df_filtered["rating"] >= min_rating) & (df_filtered["rating"] <= max_rating)]

# Adicionar coluna de sentimento
if "rating" in df_filtered.columns:
    df_filtered["sentiment"] = df_filtered["rating"].apply(classify_sentiment_advanced)

st.sidebar.success(f"✅ {len(df_filtered)} avaliações filtradas")

# ==================== TABS PRINCIPAIS ====================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visão Geral", "💬 Análise de Texto", "📈 Tendências", "📋 Qualidade"])

# ==================== TAB 1: VISÃO GERAL ====================
with tab1:
    st.header("📊 Métricas Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Avaliações", len(df_filtered))
    with col2:
        avg_rating = round(df_filtered["rating"].mean(), 2) if "rating" in df_filtered.columns else 0
        st.metric("Média das Notas", f"⭐ {avg_rating}")
    with col3:
        unique_users = df_filtered["user"].nunique() if "user" in df_filtered.columns else 0
        st.metric("Usuários Únicos", unique_users)
    with col4:
        if "sentiment" in df_filtered.columns:
            positive_pct = (df_filtered["sentiment"] == "Positivo").sum() / len(df_filtered) * 100
            st.metric("% Positivas", f"{positive_pct:.1f}%")
    
    st.divider()
    
    # Gráficos lado a lado
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Distribuição das Notas")
        if "rating" in df_filtered.columns:
            rating_counts = df_filtered["rating"].value_counts().sort_index().reset_index()
            rating_counts.columns = ["rating", "count"]
            
            bar_chart = alt.Chart(rating_counts).mark_bar(color='steelblue').encode(
                x=alt.X("rating:O", title="Nota"),
                y=alt.Y("count:Q", title="Quantidade"),
                tooltip=["rating", "count"]
            ).properties(height=300)
            
            st.altair_chart(bar_chart, use_container_width=True)
    
    with col_right:
        st.subheader("Distribuição de Sentimento")
        if "sentiment" in df_filtered.columns:
            sentiment_counts = df_filtered["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["sentiment", "count"]
            
            colors = {"Positivo": "#2ecc71", "Neutro": "#f39c12", "Negativo": "#e74c3c"}
            
            pie_chart = alt.Chart(sentiment_counts).mark_arc().encode(
                theta=alt.Theta(field="count", type="quantitative"),
                color=alt.Color(
                    field="sentiment",
                    type="nominal",
                    scale=alt.Scale(
                        domain=["Positivo", "Neutro", "Negativo"],
                        range=["#2ecc71", "#f39c12", "#e74c3c"]
                    )
                ),
                tooltip=["sentiment", "count"]
            ).properties(height=300)
            
            st.altair_chart(pie_chart, use_container_width=True)
    
    st.divider()
    
    # Tendência temporal
    st.subheader("Evolução da Média de Notas")
    if "date" in df_filtered.columns and "rating" in df_filtered.columns:
        df_filtered["month"] = df_filtered["date"].dt.to_period("M").dt.to_timestamp()
        rating_trend = df_filtered.groupby("month")["rating"].mean().reset_index()
        
        line_chart = alt.Chart(rating_trend).mark_line(point=True, color='steelblue').encode(
            x=alt.X("month:T", title="Mês"),
            y=alt.Y("rating:Q", title="Média da Nota", scale=alt.Scale(domain=[0, 5])),
            tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("rating:Q", format=".2f")]
        ).properties(height=300)
        
        st.altair_chart(line_chart, use_container_width=True)

# ==================== TAB 2: ANÁLISE DE TEXTO ====================
with tab2:
    st.header("💬 Análise de Texto e NLP")
    
    analysis_type = st.radio(
        "Escolha o tipo de análise:",
        ["Palavras Únicas", "Frases (Bigramas)", "Word Cloud"],
        horizontal=True,
        key="analysis_type_radio"
    )
    
    st.divider()
    
    for sentiment in ["Positivo", "Neutro", "Negativo"]:
        sentiment_data = df_filtered[df_filtered["sentiment"] == sentiment]
        
        if len(sentiment_data) == 0:
            continue
        
        # Escolher cor e emoji por sentimento
        emoji = "😊" if sentiment == "Positivo" else "😐" if sentiment == "Neutro" else "😞"
        color = "#2ecc71" if sentiment == "Positivo" else "#f39c12" if sentiment == "Neutro" else "#e74c3c"
        
        with st.expander(f"{emoji} {sentiment} ({len(sentiment_data)} avaliações)", expanded=True):
            
            # OPÇÃO 1: Palavras Únicas (word count)
            if analysis_type == "Palavras Únicas":
                top_words = extract_top_words(sentiment_data["clean_comment"], n_top=15)
                
                if top_words:
                    df_words = pd.DataFrame(top_words, columns=["Palavra", "Frequência"])
                    
                    chart = alt.Chart(df_words).mark_bar().encode(
                        x=alt.X("Frequência:Q"),
                        y=alt.Y("Palavra:N", sort="-x"),
                        color=alt.value(color),
                        tooltip=["Palavra", "Frequência"]
                    ).properties(height=400)
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Não há palavras suficientes para análise.")
            
            # OPÇÃO 2: Frases (Bigramas)
            elif analysis_type == "Frases (Bigramas)":
                bigrams = extract_ngrams(sentiment_data["clean_comment"], n=2, top_n=10)
                
                if bigrams:
                    df_bigrams = pd.DataFrame(bigrams, columns=["Frase", "Frequência"])
                    
                    chart = alt.Chart(df_bigrams).mark_bar().encode(
                        x=alt.X("Frequência:Q"),
                        y=alt.Y("Frase:N", sort="-x"),
                        color=alt.value(color),
                        tooltip=["Frase", "Frequência"]
                    ).properties(height=350)
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Dados insuficientes para gerar bigramas.")
            
            # OPÇÃO 3: Word Cloud
            elif analysis_type == "Word Cloud":
                text = " ".join([str(c) for c in sentiment_data["clean_comment"].dropna() if c])
                
                wordcloud_image = generate_wordcloud(text, sentiment)
                
                if wordcloud_image:
                    st.image(wordcloud_image, use_container_width=False, width=800)
                else:
                    st.info("Não há texto suficiente para gerar word cloud.")

# ==================== TAB 3: TENDÊNCIAS ====================
with tab3:
    st.header("📈 Análise Temporal e Padrões")
    
    if "date" in df_filtered.columns:
        df_filtered["day_of_week"] = df_filtered["date"].dt.day_name()
        df_filtered["hour"] = df_filtered["date"].dt.hour
        df_filtered["is_weekend"] = df_filtered["date"].dt.dayofweek >= 5
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Avaliações por Dia da Semana")
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_counts = df_filtered["day_of_week"].value_counts().reindex(dow_order, fill_value=0).reset_index()
            dow_counts.columns = ["day", "count"]
            
            chart = alt.Chart(dow_counts).mark_bar(color='steelblue').encode(
                x=alt.X("day:N", title="Dia da Semana", sort=dow_order),
                y=alt.Y("count:Q", title="Quantidade"),
                tooltip=["day", "count"]
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
        
        with col2:
            st.subheader("Rating Médio: Semana vs Fim de Semana")
            weekend_avg = df_filtered.groupby("is_weekend")["rating"].mean().reset_index()
            weekend_avg["period"] = weekend_avg["is_weekend"].map({True: "Fim de Semana", False: "Semana"})
            
            chart = alt.Chart(weekend_avg).mark_bar(color='coral').encode(
                x=alt.X("period:N", title=""),
                y=alt.Y("rating:Q", title="Rating Médio", scale=alt.Scale(domain=[0, 5])),
                tooltip=["period", alt.Tooltip("rating:Q", format=".2f")]
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
        
        st.divider()
        
        # HEATMAP FUNCIONAL
        st.subheader("📊 Heatmap: Rating Médio por Dia e Hora")
        
        # Preparar dados
        heatmap_data = df_filtered.groupby(["day_of_week", "hour"])["rating"].mean().reset_index()
        
        # Criar heatmap com Altair
        heatmap = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X("hour:O", title="Hora do Dia"),
            y=alt.Y("day_of_week:N", title="Dia da Semana", sort=dow_order),
            color=alt.Color("rating:Q", scale=alt.Scale(scheme="goldred"), title="Rating Médio"),
            tooltip=[
                alt.Tooltip("day_of_week:N", title="Dia"),
                alt.Tooltip("hour:O", title="Hora"),
                alt.Tooltip("rating:Q", format=".2f", title="Rating")
            ]
        ).properties(width=700, height=400)
        
        st.altair_chart(heatmap, use_container_width=True)

# ==================== TAB 4: QUALIDADE DOS DADOS ====================
with tab4:
    st.header("📋 Qualidade e Integridade dos Dados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        duplicates = df.duplicated().sum()
        st.metric("Duplicatas", duplicates)
    
    with col2:
        empty_comments = df["comment"].isna().sum()
        st.metric("Comentários Vazios", empty_comments)
    
    with col3:
        data_freshness = (pd.Timestamp.now() - df["date"].max()).days
        st.metric("Última Avaliação", f"{data_freshness} dias atrás")
    
    st.divider()
    
    st.subheader("Completude por Coluna")
    completeness = (1 - df.isnull().sum() / len(df)) * 100
    df_completeness = completeness.reset_index()
    df_completeness.columns = ["Coluna", "Completude (%)"]
    df_completeness["Completude (%)"] = df_completeness["Completude (%)"].round(2)
    
    chart = alt.Chart(df_completeness).mark_bar(color='teal').encode(
        x=alt.X("Completude (%):Q", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Coluna:N", sort="-x"),
        tooltip=["Coluna", "Completude (%)"]
    ).properties(height=200)
    
    st.altair_chart(chart, use_container_width=True)
    
    st.divider()
    
    # Download de dados
    st.subheader("💾 Exportar Dados")
    
    col_download1, col_download2, col_download3 = st.columns(3)
    
    with col_download1:
        # CSV com UTF-8 BOM (abre corretamente no Excel)
        csv_utf8_bom = df_filtered.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📊 CSV (UTF-8)",
            data=csv_utf8_bom,
            file_name=f"reviews_filtrados_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help=".csv"
        )
    
    with col_download2:
        # Excel (XLSX) - formato nativo
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_filtered.to_excel(writer, sheet_name='Reviews', index=False)
        
        st.download_button(
            label="📗 Excel (XLSX)",
            data=buffer.getvalue(),
            file_name=f"reviews_filtrados_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.ms-excel",
            help=".xlsx"
        )
    
    with col_download3:
        # JSON (para desenvolvedores)
        json_data = df_filtered.to_json(orient='records', force_ascii=False, indent=2)
        st.download_button(
            label="🔧 JSON",
            data=json_data,
            file_name=f"reviews_filtrados_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            help=".json"
        )