# Figure — Listing 2.15: Four-stage recommender framework
# Source: chapters/ch02.md lines 533-553
# Chapter: 2
# Category: api-drift  (executable=False, expected=skip)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Import fourstage recommender framework
#   #B import implementations
#   #C create an instance of the retriever.
#   #D create an instance of the four-stage recommender.
#   #E call recommend, to invoke the whole pipeline.
from recsys import FourStageRecommender
from recsys.retrievals import ItemKNNRetrieval
from recsys.filters import HistoryFilter
from recsys.scorers import PopularityScorer
from recsys.rankers import WeightedRanker

retrieval = ItemKNNRetrieval(ratings)
filter=HistoryFiltering(ratings)
scorer=PopularityScoring(ratings)

recommender = FourStageRecommender(
    retrieval=retrieval,
    filter=filter,
    scorer=scorer,
    ordering=WeightedRanker(weights={'similarity': 0.7, 'popularity': 0.3})
)

recommendations = recommender.recommend(
    user_id='123',
    k=10
)
