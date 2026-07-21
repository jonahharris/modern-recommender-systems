# Figure - Listing 4.7: Popularity baseline
# Source: chapters/ch04.md lines 334-355
# Chapter: 4
# Category: needs-training  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def generate_recommendations_popularity(train_df, user_ids, k=10):
    """
    Generate popularity-based recommendations.
    Returns DataFrame with [user_id, item_id, rank, score]
    """
    item_counts = train_df['item_id'].value_counts()  #A
    top_items = item_counts.head(k)  #B

    recommendations = []
    for user_id in user_ids:
        for rank, (item_id, count) in enumerate(top_items.items(), 1):
            recommendations.append({  #C
                'user_id': user_id,
                'item_id': item_id,
                'rank': rank,
                'score': count
            })

    return pd.DataFrame(recommendations)

pop_recs = generate_recommendations_popularity(train, test_user_ids, k=10)
pop_metrics = calculate_precision_at_k(pop_recs, ground_truth, k=10)

# Callout annotations (from the book):
#   #A Count interactions per item
#   #B Get top k most popular
#   #C Same recommendations for everyone
