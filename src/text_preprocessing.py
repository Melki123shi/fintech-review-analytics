from zipfile import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Fallback English stopwords (doesn't require NLTK download)
ENGLISH_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 
    'by', 'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 
    'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 
    'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 
    'me', 'might', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 
    'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'should', 
    'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 
    'then', 'there', 'these', 'they', 'this', 'those', 'to', 'too', 'under', 'until', 'up', 
    'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'with', 
    'you', 'your', 'yours', 'yourself', 'yourselves'
}

def get_stopwords():
    """Load stopwords with fallback to built-in list."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except:
        return ENGLISH_STOPWORDS

def preprocess_text(logger, text, use_lemmatization=True, nlp=None):
    """
    Preprocess text: lowercase, remove special chars, tokenize, remove stopwords, lemmatize.
    
    Parameters:
    -----------
    text : str
        Raw text to preprocess
    use_lemmatization : bool
        Whether to apply lemmatization via spaCy
    nlp : spacy Language object
        Loaded spaCy model for lemmatization
        
    Returns:
    --------
    list : List of processed tokens
    """
    if pd.isna(text):
        return []
    
    text = str(text).lower().strip()
    
    # Remove URLs, emails, special characters but keep alphanumeric and spaces
    import re
    text = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    if not tokens:
        return []
    
    # Remove stopwords
    stop_words = get_stopwords()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    # Lemmatization (if spaCy available)
    if use_lemmatization and nlp is not None:
        try:
            doc = nlp(' '.join(tokens))
            tokens = [token.lemma_ for token in doc if not token.is_punct]
        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}")
    
    return tokens



def extract_keywords_tfidf(logger, df, text_column="review", bank_column="bank", 
                           ngram_range=(1, 2), top_n=20, use_lemmatization=True, nlp=None):
    """
    Extract top keywords/n-grams per bank using TF-IDF.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with reviews
    text_column : str
        Column name for review text
    bank_column : str
        Column name for bank identifiers
    ngram_range : tuple
        (min, max) n-gram lengths
    top_n : int
        Number of top keywords to extract per bank
    use_lemmatization : bool
        Whether to lemmatize before TF-IDF
    nlp : spacy model
        Loaded spaCy model
        
    Returns:
    --------
    dict : Dictionary with bank names as keys and lists of (keyword, tfidf_score) tuples as values
    """
    keywords_by_bank = {}
    
    for bank in df[bank_column].unique():
        bank_df = df[df[bank_column] == bank]
        
        # Preprocess texts
        processed_texts = bank_df[text_column].apply(
            lambda x: ' '.join(preprocess_text(logger, x, use_lemmatization, nlp))
        ).tolist()
        
        # Remove empty texts
        processed_texts = [t for t in processed_texts if t.strip()]
        
        if not processed_texts:
            keywords_by_bank[bank] = []
            continue
        
        # Apply TF-IDF
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=top_n * 3,
            min_df=2,
            max_df=0.8
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate mean TF-IDF scores
            mean_tfidf = tfidf_matrix.mean(axis=0).A1
            
            # Get top keywords
            top_indices = mean_tfidf.argsort()[-top_n:][::-1]
            top_keywords = [
                (feature_names[i], round(mean_tfidf[i], 4)) 
                for i in top_indices
            ]
            
            keywords_by_bank[bank] = top_keywords
            logger.info(f"✓ Extracted {len(top_keywords)} keywords for {bank}")
            
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed for {bank}: {e}")
            keywords_by_bank[bank] = []
    
    return keywords_by_bank

def define_themes():
    """
    Define business-relevant themes and their associated keywords.
    
    Returns:
    --------
    dict : Theme name -> list of associated keywords
    """
    themes = {
        "Account Access & Authentication": {
            "keywords": [
                "login", "signin", "sign in", "password", "account", "authentication",
                "verification", "otp", "access", "create account", "register", "credential",
                "auth", "session", "logout", "forgot password", "2fa", "two factor",
                "verify", "confirm", "unlock"
            ]
        },
        "Transaction & Payment Performance": {
            "keywords": [
                "transfer", "payment", "slow", "fast", "delay", "timeout", "fail",
                "error", "transaction", "send money", "receive", "pending", "stuck",
                "complete", "process", "speed", "instant", "realtime", "real time",
                "successful", "unsuccessful", "retry"
            ]
        },
        "UI/UX & Design": {
            "keywords": [
                "interface", "ui", "ux", "design", "layout", "button", "navigation",
                "app", "screen", "display", "visual", "theme", "color", "font",
                "responsive", "usable", "intuitive", "confusing", "cluttered", "clean",
                "menu", "icon", "userfriendly"
            ]
        },
        "Customer Support & Service": {
            "keywords": [
                "support", "help", "customer service", "response", "issue", "complaint",
                "feedback", "assist", "resolution", "contact", "chat", "email",
                "call", "reach", "responsive", "helpful", "rude", "friendly",
                "agent", "team"
            ]
        },
        "Features & Functionality": {
            "keywords": [
                "feature", "function", "request", "missing", "update", "improve",
                "new", "capability", "tool", "option", "setting", "preference",
                "notification", "alert", "reminder", "billing", "history", "statement",
                "limit", "enhancement"
            ]
        }
    }
    
    return themes

def assign_theme(text, themes_dict, use_lemmatization=True, nlp=None):
    """
    Assign one or more themes to a review based on keyword matching.
    
    Parameters:
    -----------
    text : str
        Review text
    themes_dict : dict
        Theme definitions with keywords
    use_lemmatization : bool
        Whether to lemmatize before matching
    nlp : spacy model
        Loaded spaCy model
        
    Returns:
    --------
    str : Assigned theme name (or "Unclassified" if no match)
    """
    if pd.isna(text):
        return "Unclassified"
    
    # Preprocess text
    processed_tokens = preprocess_text(text, use_lemmatization, nlp)
    processed_text = ' '.join(processed_tokens).lower()
    
    # Count keyword matches per theme
    theme_scores = {}
    for theme_name, theme_info in themes_dict.items():
        keywords = theme_info["keywords"]
        match_count = sum(1 for kw in keywords if kw in processed_text)
        theme_scores[theme_name] = match_count
    
    # Return best matching theme
    if max(theme_scores.values()) > 0:
        best_theme = max(theme_scores, key=theme_scores.get)
        return best_theme
    else:
        return "Unclassified"

def assign_themes_to_dataframe(df, themes_dict, text_column="review", 
                               use_lemmatization=True, nlp=None):
    """
    Apply theme assignment to all reviews in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with reviews
    themes_dict : dict
        Theme definitions
    text_column : str
        Column name for review text
    use_lemmatization : bool
        Whether to lemmatize
    nlp : spacy model
        Loaded spaCy model
        
    Returns:
    --------
    pd.DataFrame : DataFrame with new 'identified_theme' column
    """
    df = df.copy()
    df["identified_theme"] = df[text_column].apply(
        lambda x: assign_theme(x, themes_dict, use_lemmatization, nlp)
    )
    return df

def prepare_results_dataframe(logger, df, output_columns=None):
    """
    Prepare final results DataFrame with required columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with all analysis results
    output_columns : list
        Columns to include in output
        
    Returns:
    --------
    pd.DataFrame : Results DataFrame
    """
    if output_columns is None:
        output_columns = [
            'review_id', 'review', 'bank', 'rating',
            'sentiment', 'sentiment_confidence', 'identified_theme'
        ]
    
    # Ensure all columns exist
    for col in output_columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
    
    # Select available columns
    available_cols = [c for c in output_columns if c in df.columns]
    results_df = df[available_cols].copy()
    
    # Rename for clarity
    results_df = results_df.rename(columns={
        'review': 'review_text',
        'sentiment': 'sentiment_label',
        'sentiment_confidence': 'sentiment_score'
    })
    
    return results_df

def save_results_to_csv(logger, df, output_path='data/processed/sentiment_and_themes_analysis.csv'):
    """
    Save analysis results to CSV file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Results DataFrame
    output_path : str
        Path to save CSV file
        
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Results saved to {output_path}")
        print(f"✓ Results saved: {output_path}")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to save results: {e}")
        return False

