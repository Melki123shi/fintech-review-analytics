import re
import pandas as pd
from google_play_scraper import app, reviews, Sort
import os


def clean_review_text(text):
    """
    Clean review text by:
    - Converting to lowercase
    - Removing URLs
    - Removing extra whitespace
    - Removing special characters (keeping only alphanumeric, spaces, and basic punctuation)
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove emojis and special Unicode characters
    text = text.encode("ascii", "ignore").decode("ascii")

    return text


def display_app_info(app_id):
    """
    Fetch and display app information from Google Play Store.
    """
    app_info = app(app_id, lang="en", country="et")

    print("=" * 50)
    print(f"{app_info['title']} App Info")
    print("=" * 50)
    print(f"App Title   : {app_info['title']}")
    print(f"Current Score: {app_info['score']}")
    print(f"Total Ratings: {app_info['ratings']:,}")
    print(f"Total Reviews: {app_info['reviews']:,}")
    print(f"Installs     : {app_info['installs']}")

def review_dataframe(reviews, app_info):
    ""
    raw_data = []

    for r in reviews:
        raw_data.append(
            {
                "review_id": r.get("reviewId", ""),
                "review": r.get("content", ""),
                "rating": r.get("score", None),
                "date": r.get("at", None),
                "bank": app_info['title'],
                "source": "Google Play",
            }
        )

    # Build a DataFrame
    df = pd.DataFrame(raw_data)
    return df


def remove_duplicates(df):
    """
    Remove duplicate reviews based on 'review_id'.
    """
    initial_count = len(df)
    df = df.drop_duplicates(subset="review_id")
    final_count = len(df)
    print(f"Removed {initial_count - final_count} duplicate reviews")
    print(f"Remaining: {len(df)} reviews")
    return df


def handle_missing_data(df):
    """
    Handle missing data by dropping rows with missing critical columns.
    """
    initial_count = len(df)
    critical_cols = ["review", "rating"]
    df = df.dropna(subset=critical_cols)
    final_count = len(df)
    print(f"Removed {initial_count - final_count} rows with missing critical data")
    print(f"Remaining: {len(df)} reviews")

    return df


def normalize_dates(df):
    """
    Normalize date formats to ISO 8601 (YYYY-MM-DD).
    """
    print("Before normalization:")
    print(df["date"].head(3).to_string())
    print(f"dtype: {df['date'].dtype}")

    # Convert to pandas datetime, then format as YYYY-MM-DD string
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    print("\nAfter normalization:")
    print(df["date"].head(3).to_string())
    print(f"dtype: {df['date'].dtype}")

    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")

    return df


def validate_rating(df):
    """
    Validate that ratings are integers between 1 and 5.
    """
    invalid_ratings = df[~df["rating"].between(1, 5)]
    if not invalid_ratings.empty:
        print("Found invalid ratings:")
        print(invalid_ratings[["review_id", "rating"]])
        # Optionally, drop or correct these rows
        df = df[df["rating"].between(1, 5)]
        print(f"Removed {len(invalid_ratings)} reviews with invalid ratings")
    else:
        print("All ratings are valid (1-5).")

    print(f"Remaining: {len(df)} reviews")

    return df


def rating_distribution(df):
    """
    Print the distribution of ratings in a simple text-based format.
    """
    print("Rating distribution:")
    rating_counts = df["rating"].value_counts().sort_index(ascending=False)
    for rating, count in rating_counts.items():
        bar = "█" * (count // 5)  # Scale the bar length
        print(f"  {int(rating)} stars: {count:>4}  {bar}")


def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Saved to: {output_path}")


def preprocessing_report(df_raw, df_clean):
    print("=" * 55)
    print("  PREPROCESSING REPORT — Awash Bank Reviews")
    print("=" * 55)

    original_count = len(df_raw)
    final_count = len(df_clean)
    removed_total = original_count - final_count
    retention_rate = final_count / original_count * 100

    print(f"\n  Raw reviews collected  : {original_count:>6}")
    print(f"  Reviews after cleaning : {final_count:>6}")
    print(f"  Reviews removed        : {removed_total:>6}")
    print(f"  Data retention rate    : {retention_rate:>5.1f}%")

    quality = (
        "EXCELLENT"
        if retention_rate >= 95
        else ("GOOD" if retention_rate >= 90 else "NEEDS ATTENTION")
    )
    print(f"  Data quality           : {quality}")

    print(f"\n  Date range : {df_clean['date'].min()}  to  {df_clean['date'].max()}")

    rating_distribution(df_clean)

    print("\n  Text length stats:")
    lengths = df_clean["review"].str.len()
    print(f"    Min    : {lengths.min()} characters")
    print(f"    Median : {lengths.median():.0f} characters")
    print(f"    Max    : {lengths.max()} characters")

    print("\n  Columns in final CSV:")
    for col in df_clean.columns:
        print(f"    - {col}")

    print("\n" + "=" * 55)
