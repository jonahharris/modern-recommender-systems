# Figure - Listing 2.15: Four-stage recommender framework
# Source: chapters/ch02.md lines 533-553
# Chapter: 2
# Category: api-drift  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
from recsys.fourstage_recsys.pipeline import FourStageRecommender  #A
from recsys.fourstage_recsys.retrieval.itemknn_retrieval import ItemKNNRetrieval  #B
from recsys.fourstage_recsys.filtering.history_filtering import HistoryFiltering  #B
from recsys.fourstage_recsys.scoring.popularity import PopularityScoring  #B
from recsys.fourstage_recsys.ordering.weighted_ranker import WeightedRanker  #B
from recsys.fourstage_recsys.recsys_context import RecommendationContext  #B

retrieval = ItemKNNRetrieval(ratings)  #C
filter = HistoryFiltering(ratings)  #C
scorer = PopularityScoring(ratings)  #C

recommender = FourStageRecommender(
    retrieval=retrieval,
    filter=filter,
    scorer=scorer,
    ordering=WeightedRanker(weights={'similarity': 0.7, 'popularity': 0.3})
)  #D

recommendations = recommender.recommend(
    RecommendationContext(user_id='123', seed_items=['1'], k=10)
)  #E

# Callout annotations (from the book):
#   #A Import the four-stage recommender framework
#   #B Import the stage implementations
#   #C Create the retrieval, filtering and scoring stages
#   #D Assemble the four-stage recommender
#   #E Call recommend to run the whole pipeline
