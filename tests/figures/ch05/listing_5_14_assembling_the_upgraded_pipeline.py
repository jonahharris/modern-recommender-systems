# Figure — Listing 5.14: Assembling the upgraded pipeline
# Source: chapters/ch05.md lines 852-878
# Chapter: 5
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A ANN retrieval with FAISS (milliseconds, not seconds)
#   #B Cross-encoder scoring for precise relevance estimates
#   #C Same pipeline interface as Chapter 2
#   #D Returns the top 10 recommendations
retrieval = ANNRetrieval(
  index=hnsw_index,
  item_embeddings=item_embeddings,
  id_to_item=id_to_item,
  item_to_id=item_to_id,
)

scorer = CrossEncoderScoring(
  cross_encoder=trained_cross_encoder,
  user_features=user_feature_store,
  item_features=item_feature_store,
)

recommender = FourStageRecommender(
  retrieval=retrieval,
  filter=DeduplicationFilter(),
  scorer=scorer,
  ranker=ScoreRanker(),
)

recs = recommender.recommend(
  RecommendationContext(
    seed_movie_id="pulp_fiction",
    user_id="user_42",
    k=10,
  )
)
