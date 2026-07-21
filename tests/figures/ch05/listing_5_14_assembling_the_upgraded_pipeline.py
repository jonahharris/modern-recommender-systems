# Figure - Listing 5.14: Assembling the upgraded pipeline
# Source: chapters/ch05.md lines 852-878
# Chapter: 5
# Category: api-drift  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
from recsys.fourstage_recsys.filtering.history_filtering import HistoryFiltering
from recsys.fourstage_recsys.ordering.weighted_ranker import WeightedRanker
from recsys.fourstage_recsys.recsys_context import RecommendationContext

retrieval = ANNRetrieval(
  index=hnsw_index,
  item_embeddings=item_embeddings,
  id_to_item=id_to_item,
  item_to_id=item_to_id,
)  #A

scorer = CrossEncoderScoring(
  cross_encoder=trained_cross_encoder,
  user_features=user_feature_store,
  item_features=item_feature_store,
)  #B

recommender = FourStageRecommender(
  retrieval=retrieval,
  filter=HistoryFiltering(ratings),
  scorer=scorer,
  ordering=WeightedRanker(weights={"relevance": 1.0}),
)  #C

recs = recommender.recommend(
  RecommendationContext(
    seed_items=["pulp_fiction"],
    user_id="user_42",
    k=10,
  )
)  #D

# Callout annotations (from the book):
#   #A ANN retrieval with FAISS (milliseconds, not seconds)
#   #B Cross-encoder scoring for precise relevance estimates
#   #C Same pipeline interface as Chapter 2
#   #D Returns the top 10 recommendations
