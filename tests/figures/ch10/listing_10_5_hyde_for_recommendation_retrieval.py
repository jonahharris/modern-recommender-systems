# Figure — Listing 10.5: HyDE for recommendation retrieval
# Source: chapters/ch10.md lines 252-263
# Chapter: 10
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A The LLM generates a description, not a title — this becomes the search query
#   #B The hypothetical description is embedded and used for vector search
def hyde_retrieval(llm, retriever, user_query, k=10):
    """Retrieve using a hypothetical item description."""
    hypothetical = llm.generate(
        system_prompt="You are a movie database. Given a "
            "user's request, write a detailed description "
            "of a movie that would perfectly match. Include "
            "genre, tone, themes, and plot elements. The "
            "movie does not need to be real.",
        user_message=user_query
    )
    results = retriever.search(hypothetical, k=k)
    return results, hypothetical
