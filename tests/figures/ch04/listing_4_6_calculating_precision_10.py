# Figure - Listing 4.6: Calculating precision@10
# Source: chapters/ch04.md lines 288-312
# Chapter: 4
# Category: needs-training  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def calculate_precision_at_k(recommendations_df, relevance_set_df, k=10):
    """
    Calculate precision@k for each user.
    Returns DataFrame with [user_id, precision, num_hits]
    """
    top_k_recs = recommendations_df[recommendations_df['rank'] <= k]  #A

    merged = top_k_recs.merge(  #B
        relevance_set_df,
        on=['user_id', 'item_id'],
        how='left',
        indicator=True
    )
    merged['is_hit'] = (merged['_merge'] == 'both').astype(int)  #C

    user_metrics = merged.groupby('user_id').agg(  #D
        num_hits=('is_hit', 'sum'),
        num_recs=('is_hit', 'count')
    ).reset_index()

    user_metrics['precision'] = user_metrics['num_hits'] / user_metrics['num_recs']  #E

    return user_metrics

als_metrics = calculate_precision_at_k(als_recs, relevance_set, k=10)

# Callout annotations (from the book):
#   #A Keep only the top k items per user
#   #B Join recommendations with ground truth
#   #C Mark hits (items in both)
#   #D Count hits per user
#   #E Calculate precision \= hits / k
