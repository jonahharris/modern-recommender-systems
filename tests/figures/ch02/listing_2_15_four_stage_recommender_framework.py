# Figure - Listing 2.15: Four-stage recommender framework
# Source: chapters/ch02.md lines 533-553
# Chapter: 2
# Category: api-drift  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
from recsys import FourStageRecommender  #A
from recsys.retrievals import ItemKNNRetrieval  #B
from recsys.filters import HistoryFilter  #B
from recsys.scorers import PopularityScorer  #B
from recsys.rankers import WeightedRanker  #B

retrieval = ItemKNNRetrieval(ratings)  #C
filter=HistoryFiltering(ratings)  #D
scorer=PopularityScoring(ratings)  #E

recommender = FourStageRecommender(
    retrieval=retrieval,  #C,
    filter=filter,
    scorer=scorer,
    ordering=WeightedRanker(weights={'similarity': 0.7, 'popularity': 0.3})
)  #D

recommendations = recommender.recommend(
    user_id='123',
    k=10
)  #E

# Callout annotations (from the book):
#   #A Import fourstage recommender framework
#   #B import implementations
#   #C create an instance of the retriever.
#   #D create an instance of the four-stage recommender.
#   #E call recommend, to invoke the whole pipeline.
