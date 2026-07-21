# Figure - Listing 4.5: Building relevance set from test data
# Source: chapters/ch04.md lines 265-273
# Chapter: 4
# Category: needs-training  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def build_relevance_set(test_df, rating_threshold=4.0):
    """
    Extract what users actually liked in the test period.
    Returns DataFrame with columns [user_id, item_id]
    """
    relevant = test_df[test_df['rating'] >= rating_threshold].copy()  #A
    return relevant[['user_id', 'item_id']]  #B

relevance_set = build_relevance_set(test, rating_threshold=4.0)

# Callout annotations (from the book):
#   #A filter for high ratings only
#   #B Keep just user and item columns
