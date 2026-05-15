import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# ---------------------------------------------------
# Load Sentiment Model
# ---------------------------------------------------

# DistilBERT model fine-tuned on SST-2
# Chosen because:
# - Fast transformer model
# - Good sentiment accuracy
# - Pretrained and easy to use
# - Better contextual understanding than rule-based methods like VADER

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Enable progress bar
tqdm.pandas()


# ---------------------------------------------------
# Sentiment Prediction Function
# ---------------------------------------------------

def analyze_sentiment(text):

    # Handle empty values
    if pd.isna(text) or str(text).strip() == "":
        return {
            "sentiment": "neutral",
            "confidence": 0.0
        }

    try:
        result = sentiment_pipeline(str(text))[0]

        label = result["label"]
        score = result["score"]

        # Convert labels
        if label == "POSITIVE":
            sentiment = "positive"
        elif label == "NEGATIVE":
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "confidence": round(score, 4)
        }

    except Exception:
        return {
            "sentiment": "neutral",
            "confidence": 0.0
        }


# ---------------------------------------------------
# Apply Sentiment Analysis
# ---------------------------------------------------

def apply_sentiment_analysis(df, review_column="review"):

    # Analyze each review
    results = df[review_column].progress_apply(analyze_sentiment)

    # Create new columns
    df["sentiment"] = results.apply(lambda x: x["sentiment"])
    df["sentiment_confidence"] = results.apply(
        lambda x: x["confidence"]
    )

    return df


# ---------------------------------------------------
# Aggregate by Bank
# ---------------------------------------------------

def aggregate_sentiment_by_bank(
    df,
    bank_column="bank"
):

    aggregation = (
        df.groupby(bank_column)
        .agg(
            total_reviews=("sentiment", "count"),
            mean_confidence=("sentiment_confidence", "mean")
        )
        .reset_index()
    )

    return aggregation


# ---------------------------------------------------
# Aggregate by Star Rating
# ---------------------------------------------------

def aggregate_sentiment_by_rating(
    df,
    rating_column="rating"
):

    aggregation = (
        df.groupby(rating_column)
        .agg(
            total_reviews=("sentiment", "count"),
            mean_confidence=("sentiment_confidence", "mean")
        )
        .reset_index()
        .sort_values(rating_column)
    )

    return aggregation


# ---------------------------------------------------
# Sentiment Distribution Tables
# ---------------------------------------------------

def sentiment_distribution_by_bank(
    df,
    bank_column="bank"
):

    return pd.crosstab(
        df[bank_column],
        df["sentiment"]
    )


def sentiment_distribution_by_rating(
    df,
    rating_column="rating"
):

    return pd.crosstab(
        df[rating_column],
        df["sentiment"]
    )


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------

# Apply sentiment analysis
# df = apply_sentiment_analysis(df)

# # View sample results
# print(df[
#     [
#         "review",
#         "sentiment",
#         "sentiment_confidence"
#     ]
# ].head())

# # Aggregate by bank
# bank_summary = aggregate_sentiment_by_bank(df)

# print("\nSentiment Summary by Bank")
# print(bank_summary)

# # Aggregate by rating
# rating_summary = aggregate_sentiment_by_rating(df)

# print("\nSentiment Summary by Rating")
# print(rating_summary)

# # Sentiment distribution by bank
# print("\nSentiment Distribution by Bank")
# print(sentiment_distribution_by_bank(df))

# # Sentiment distribution by rating
# print("\nSentiment Distribution by Rating")
# print(sentiment_distribution_by_rating(df))

