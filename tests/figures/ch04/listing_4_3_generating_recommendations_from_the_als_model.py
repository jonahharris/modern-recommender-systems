# Figure — Listing 4.3: Generating recommendations from the ALS model
# Source: chapters/ch04.md lines 223-253
# Chapter: 4
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Loop through each user
#   #B Get recommendations from ALS model
#   #C Store with rank and score
import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares

def generate_recommendations_als(model, user_item_matrix, user_ids, k=10):
    """
    Generate recommendations for a list of users.
    Returns DataFrame with columns [user_id, item_id, rank, score]
    """
    all_recommendations = []

    for user_id in user_ids:
        item_ids, scores = model.recommend(
            user_id,
            user_item_matrix[user_id],
            N=k
        )
        for rank, (item_id, score) in enumerate(zip(item_ids, scores), 1):
            all_recommendations.append({
                'user_id': user_id,
                'item_id': item_id,
                'rank': rank,
                'score': score
            })

    return pd.DataFrame(all_recommendations)

test_user_ids = test['user_id'].unique()
als_recs = generate_recommendations_als(
    als_model, user_item_matrix, test_user_ids, k=10
)
