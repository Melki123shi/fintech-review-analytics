from google_play_scraper import reviews, Sort
def scrap_reviews(app_id, num_reviews=700):
    """
    Scrape reviews for a given app ID from Google Play Store.
    """
    print(f"Scraping reviews for {app_id}...")

    result, continuation_token = reviews(
        app_id,
        lang="en",
        country="et",
        sort=Sort.NEWEST,  # Most recent first
        count=num_reviews,  # Ask for more than 400 to be safe
        filter_score_with=None,  # All star ratings
    )

    print(f"Collected {len(result)} raw reviews")
    return result
