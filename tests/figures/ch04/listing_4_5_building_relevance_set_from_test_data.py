# Figure — Listing 4.5: Building relevance set from test data
# Source: chapters/ch04.md lines 265-273
# Chapter: 4
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A filter for high ratings only
#   #B Keep just user and item columns
def build_relevance_set(test_df, rating_threshold=4.0):
    """
    Extract what users actually liked in the test period.
    Returns DataFrame with columns [user_id, item_id]
    """
    relevant = test_df[test_df['rating'] >= rating_threshold].copy()
    return relevant[['user_id', 'item_id']]

relevance_set = build_relevance_set(test, rating_threshold=4.0)
