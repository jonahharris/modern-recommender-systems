# Figure - Listing 10.7: End-to-end RAG recommendation
# Source: chapters/ch10.md lines 328-340
# Chapter: 10
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def recommend_with_rag(query, retriever, llm,
                       user_id=None, **kwargs):
    candidates = retriever.hybrid_search(  #A
        query, user_id=user_id, k=15
    )
    system, user_msg = build_recommendation_prompt(
        query, candidates, **kwargs
    )
    response = llm.generate(  #B
        system_prompt=system,
        user_message=user_msg
    )
    return response, candidates

# Callout annotations (from the book):
#   #A Retrieve 15 candidates
#   #B The LLM selects, ranks, and explains
