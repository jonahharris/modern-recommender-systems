# Figure — Listing 4.7: Popularity baseline
# Source: chapters/ch04.md lines 334-355
# Chapter: 4
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Count interactions per item
#   #B Get top k most popular
#   #C Same recommendations for everyone
def generate_recommendations_popularity(train_df, user_ids, k=10):
    """
    Generate popularity-based recommendations.
    Returns DataFrame with [user_id, item_id, rank, score]
    """
    item_counts = train_df['item_id'].value_counts()
    top_items = item_counts.head(k)

    recommendations = []
    for user_id in user_ids:
        for rank, (item_id, count) in enumerate(top_items.items(), 1):
            recommendations.append({
                'user_id': user_id,
                'item_id': item_id,
                'rank': rank,
                'score': count
            })

    return pd.DataFrame(recommendations)

pop_recs = generate_recommendations_popularity(train, test_user_ids, k=10)
pop_metrics = calculate_precision_at_k(pop_recs, ground_truth, k=10)
