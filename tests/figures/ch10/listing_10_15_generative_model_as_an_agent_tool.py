# Figure - Listing 10.15: Generative model as an agent tool
# Source: chapters/ch10.md lines 635-664
# Chapter: 10
# Category: standalone  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
# The generative semantic-ID model, decoder, formatter, and generation
# function are the Chapter 7/9 components. They live in the chapter
# notebooks (not the installed recsys package), so the caller injects
# them rather than importing from a recsys.generative module.
class GenerativeRetrieverTool:
  def __init__(self, model, tokenizer, decoder,
               formatter, generate_fn):  #A
    self.model = model
    self.tokenizer = tokenizer
    self.decoder = decoder
    self.formatter = formatter
    self.generate_fn = generate_fn

  def recommend_from_history(self, user_record,
                              num_items=10):
    """Generate sequential recommendations."""
    formatted = self.formatter.format(  #B
      user_record, is_training=False
    )
    raw_tokens = self.generate_fn(  #C
      self.model, self.tokenizer,
      formatted, num_items=num_items
    )
    decoded = self.decoder.decode(raw_tokens)  #D
    return [
      {"title": item["title"],
       "source": "generative",
       "movie_id": item["movie_id"]}
      for item in decoded
    ]

# Callout annotations (from the book):
#   #A Inject the Chapter 7/9 generative model, tokenizer, decoder, formatter, and generation function
#   #B Format the user's history using the same formatter classes from Chapter 7
#   #C Generate semantic ID tokens autoregressively
#   #D Decode tokens back to catalog items via the SemanticDecoder
