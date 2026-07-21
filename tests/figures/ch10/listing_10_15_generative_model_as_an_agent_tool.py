# Figure - Listing 10.15: Generative model as an agent tool
# Source: chapters/ch10.md lines 635-664
# Chapter: 10
# Category: api-drift  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
from recsys.generative import (  #A
  generate_recommendations,
  SemanticDecoder
)

class GenerativeRetrieverTool:
  def __init__(self, model, tokenizer, decoder,
               formatter):
    self.model = model
    self.tokenizer = tokenizer
    self.decoder = decoder
    self.formatter = formatter

  def recommend_from_history(self, user_record,
                              num_items=10):
    """Generate sequential recommendations."""
    formatted = self.formatter.format(  #B
      user_record, is_training=False
    )
    raw_tokens = generate_recommendations(  #C
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
#   #A Import the components built in Chapter 7
#   #B Format the user's history using the same formatter classes from Chapter 7
#   #C Generate semantic ID tokens autoregressively
#   #D Decode tokens back to catalog items via SemanticDecoder
