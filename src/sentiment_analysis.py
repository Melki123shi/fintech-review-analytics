
import spacy
import nltk
from tqdm import tqdm
import pandas as pd
from transformers import pipeline
from nltk.tokenize import word_tokenize


tqdm.pandas()


def explore_data(banks_data):
    """
    Display basic statistics and sample reviews for each bank.
    
    Parameters:
    -----------
    banks_data : dict
        Dictionary with bank DataFrames
    """
    for bank_name, df in banks_data.items():
        print(f"\n{'='*60}")
        print(f"📋 {bank_name} - Data Overview")
        print(f"{'='*60}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nMissing Values:\n{df.isnull().sum()}")
        print(f"\n✓ Sample Reviews (first 3):")
        for idx, row in df.head(3).iterrows():
            print(f"  [{idx}] Rating: {row.get('rating', 'N/A')} | {row.get('review', 'N/A')[:80]}...")


def initialize_sentiment_model(logger, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Initialize DistilBERT sentiment analysis model with error handling.
    
    Parameters:
    -----------
    model_name : str
        HuggingFace model identifier
        
    Returns:
    --------
    pipeline : Transformer pipeline or None
        Loaded sentiment pipeline
    """
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1  # Use CPU, set to 0 for GPU if available
        )
        logger.info(f"✓ Successfully loaded {model_name}")
        return sentiment_pipeline
    except Exception as e:
        logger.error(f"✗ Failed to load transformer model: {e}")
        logger.info("⚠ Falling back to VADER sentiment analysis")
        return None

def download_nltk_resources(logger):
    """
    Download required NLTK resources for tokenization and sentiment analysis.
    """
    from zipfile import BadZipFile
    resources = ['punkt', 'vader_lexicon', 'averaged_perceptron_tagger']
    
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            logger.info(f"✓ {resource} already available")
        except (LookupError, BadZipFile):
            logger.info(f"Downloading {resource}...")
            try:
                nltk.download(resource, quiet=False, raise_errors=True)
                logger.info(f"✓ {resource} downloaded successfully")
            except Exception as e:
                logger.warning(f"Failed to download {resource}: {e}")
    
    logger.info("✓ NLTK resources ready")

def load_spacy_model(logger):
    """
    Load spaCy English model for advanced NLP tasks.
    
    Returns:
    --------
    nlp : spacy Language object
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("✓ spaCy model loaded")
        return nlp
    except OSError:
        logger.warning("! spaCy model not found. Run: python -m spacy download en_core_web_sm")
        return None


def analyze_sentiment(logger, text, sentiment_pipeline=None, max_length=512):
    """
    Classify sentiment of a single review text.
    
    Parameters:
    -----------
    text : str
        Review text to analyze
    sentiment_pipeline : pipeline
        DistilBERT sentiment pipeline
    max_length : int
        Maximum text length for transformer model
        
    Returns:
    --------
    dict : Contains 'sentiment' (positive/negative/neutral) and 'confidence' (0-1)
    """
    # Handle empty or null values
    if pd.isna(text) or str(text).strip() == "":
        return {
            "sentiment": "neutral",
            "confidence": 0.0
        }
    
    try:
        text_str = str(text).strip()
        
        # Truncate to max_length to avoid tokenization errors
        if len(text_str) > max_length:
            text_str = text_str[:max_length]
        
        if sentiment_pipeline is not None:
            # Use DistilBERT transformer
            result = sentiment_pipeline(text_str)[0]
            label = result["label"]
            score = result["score"]
            
            # Normalize label
            if label == "POSITIVE":
                sentiment = "positive"
            elif label == "NEGATIVE":
                sentiment = "negative"
            else:
                sentiment = "neutral"
        else:
            # Fallback: simple heuristic based on keywords
            sentiment = "neutral"
            score = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": round(float(score), 4)
        }
        
    except Exception as e:
        logger.warning(f"Error analyzing text: {e}")
        return {
            "sentiment": "neutral",
            "confidence": 0.0
        }

def apply_sentiment_analysis(logger, df, sentiment_pipeline, review_column="review", show_progress=True):
    """
    Apply sentiment analysis to all reviews in a DataFrame.
    
    Parameters:
    -----------
    logger : logging.Logger
        Logger instance for logging messages
    df : pd.DataFrame
        Input DataFrame with reviews
    sentiment_pipeline : pipeline
        DistilBERT sentiment pipeline
    review_column : str
        Name of column containing review text
    show_progress : bool
        Whether to show progress bar
        
    Returns:
    --------
    pd.DataFrame : DataFrame with new 'sentiment' and 'sentiment_confidence' columns
    """
    df = df.copy()
    
    # Apply sentiment analysis
    if show_progress:
        tqdm.pandas(desc=f"Analyzing sentiment for {df['bank']}")
        results = df[review_column].progress_apply(
            lambda x: analyze_sentiment(logger, x, sentiment_pipeline)
        )
    else:
        results = df[review_column].apply(
            lambda x: analyze_sentiment(logger, x, sentiment_pipeline)
        )
    
    # Extract sentiment and confidence
    df["sentiment"] = results.apply(lambda x: x["sentiment"])
    df["sentiment_confidence"] = results.apply(lambda x: x["confidence"])
    
    return df


def aggregate_sentiment_by_bank(df, bank_column="bank", sentiment_column="sentiment"):
    """
    Aggregate sentiment statistics by bank.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sentiment analysis results
    bank_column : str
        Column name for bank identifiers
    sentiment_column : str
        Column name for sentiment labels
        
    Returns:
    --------
    pd.DataFrame : Aggregated statistics by bank
    """
    aggregation = (
        df.groupby(bank_column)
        .agg(
            total_reviews=("sentiment", "count"),
            positive_count=("sentiment", lambda x: (x == "positive").sum()),
            negative_count=("sentiment", lambda x: (x == "negative").sum()),
            neutral_count=("sentiment", lambda x: (x == "neutral").sum()),
            mean_confidence=("sentiment_confidence", "mean"),
            min_confidence=("sentiment_confidence", "min"),
            max_confidence=("sentiment_confidence", "max")
        )
        .reset_index()
    )
    
    # Calculate percentages
    aggregation['positive_pct'] = (aggregation['positive_count'] / aggregation['total_reviews'] * 100).round(2)
    aggregation['negative_pct'] = (aggregation['negative_count'] / aggregation['total_reviews'] * 100).round(2)
    aggregation['neutral_pct'] = (aggregation['neutral_count'] / aggregation['total_reviews'] * 100).round(2)
    
    return aggregation.sort_values('total_reviews', ascending=False)

def aggregate_sentiment_by_rating(df, rating_column="rating"):
    """
    Aggregate sentiment statistics by star rating.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sentiment analysis results
    rating_column : str
        Column name for star ratings
        
    Returns:
    --------
    pd.DataFrame : Aggregated statistics by rating
    """
    aggregation = (
        df.groupby(rating_column)
        .agg(
            total_reviews=("sentiment", "count"),
            positive_count=("sentiment", lambda x: (x == "positive").sum()),
            negative_count=("sentiment", lambda x: (x == "negative").sum()),
            neutral_count=("sentiment", lambda x: (x == "neutral").sum()),
            mean_confidence=("sentiment_confidence", "mean")
        )
        .reset_index()
        .sort_values(rating_column)
    )
    
    # Calculate percentages
    aggregation['positive_pct'] = (aggregation['positive_count'] / aggregation['total_reviews'] * 100).round(2)
    aggregation['negative_pct'] = (aggregation['negative_count'] / aggregation['total_reviews'] * 100).round(2)
    aggregation['neutral_pct'] = (aggregation['neutral_count'] / aggregation['total_reviews'] * 100).round(2)
    
    return aggregation

def sentiment_distribution_by_bank(df, bank_column="bank"):
    """
    Create cross-tabulation of sentiment labels by bank.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sentiment results
    bank_column : str
        Column name for banks
        
    Returns:
    --------
    pd.DataFrame : Crosstab of sentiment by bank
    """
    return pd.crosstab(df[bank_column], df["sentiment"], margins=True)

def sentiment_distribution_by_rating(df, rating_column="rating"):
    """
    Create cross-tabulation of sentiment labels by star rating.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sentiment results
    rating_column : str
        Column name for ratings
        
    Returns:
    --------
    pd.DataFrame : Crosstab of sentiment by rating
    """
    return pd.crosstab(df[rating_column], df["sentiment"], margins=True)

def generate_kpi_report(df):
    """
    Generate Key Performance Indicator (KPI) report.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Analysis results DataFrame
        
    Returns:
    --------
    dict : Dictionary with KPI metrics
    """
    total_reviews = len(df)
    reviews_with_sentiment = (df['sentiment_score'] > 0).sum()
    sentiment_coverage = (reviews_with_sentiment / total_reviews * 100) if total_reviews > 0 else 0
    
    kpis = {
        "Total Reviews": total_reviews,
        "Sentiment Coverage (%)": round(sentiment_coverage, 2),
        "Distinct Banks": df['bank'].nunique(),
        "Distinct Themes": df['identified_theme'].nunique(),
        "Positive Reviews": (df['sentiment_label'] == 'positive').sum(),
        "Negative Reviews": (df['sentiment_label'] == 'negative').sum(),
        "Neutral Reviews": (df['sentiment_label'] == 'neutral').sum(),
        "Mean Sentiment Confidence": round(df['sentiment_score'].mean(), 4),
        "Avg Star Rating": round(df['rating'].mean(), 2) if 'rating' in df.columns else None,
    }
    
    return kpis

def generate_theme_summary(df):
    """
    Generate summary of themes by bank and sentiment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Analysis results DataFrame
        
    Returns:
    --------
    dict : Summary statistics for each theme
    """
    theme_summary = {}
    
    for theme in df['identified_theme'].unique():
        theme_df = df[df['identified_theme'] == theme]
        theme_summary[theme] = {
            'count': len(theme_df),
            'avg_rating': theme_df['rating'].mean() if 'rating' in theme_df.columns else None,
            'positive_pct': (theme_df['sentiment_label'] == 'positive').sum() / len(theme_df) * 100,
            'negative_pct': (theme_df['sentiment_label'] == 'negative').sum() / len(theme_df) * 100,
            'neutral_pct': (theme_df['sentiment_label'] == 'neutral').sum() / len(theme_df) * 100,
        }
    
    return theme_summary

