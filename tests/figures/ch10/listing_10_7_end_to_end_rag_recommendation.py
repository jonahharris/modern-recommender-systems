# Figure — Listing 10.7: End-to-end RAG recommendation
# Source: chapters/ch10.md lines 328-340
# Chapter: 10
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Retrieve 15 candidates
#   #B The LLM selects, ranks, and explains
def recommend_with_rag(query, retriever, llm,
                       user_id=None, **kwargs):
    candidates = retriever.hybrid_search(
        query, user_id=user_id, k=15
    )
    system, user_msg = build_recommendation_prompt(
        query, candidates, **kwargs
    )
    response = llm.generate(
        system_prompt=system,
        user_message=user_msg
    )
    return response, candidates
