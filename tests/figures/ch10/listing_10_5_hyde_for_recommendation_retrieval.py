# Figure - Listing 10.5: HyDE for recommendation retrieval
# Source: chapters/ch10.md lines 252-263
# Chapter: 10
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def hyde_retrieval(llm, retriever, user_query, k=10):
    """Retrieve using a hypothetical item description."""
    hypothetical = llm.generate(  #A
        system_prompt="You are a movie database. Given a "
            "user's request, write a detailed description "
            "of a movie that would perfectly match. Include "
            "genre, tone, themes, and plot elements. The "
            "movie does not need to be real.",
        user_message=user_query
    )
    results = retriever.search(hypothetical, k=k)  #B
    return results, hypothetical

# Callout annotations (from the book):
#   #A The LLM generates a description, not a title — this becomes the search query
#   #B The hypothetical description is embedded and used for vector search
