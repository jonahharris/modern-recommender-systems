# Figure - Listing 7.7: Load model and extend vocabulary
# Source: chapters/ch09.md lines 236-241
# Chapter: 9
# Category: needs-training  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)  #A
model = GPT2LMHeadModel.from_pretrained(model_name)  #B

tokenizer.add_tokens(list(new_tokens))  #C
model.resize_token_embeddings(len(tokenizer))  #D

# Callout annotations (from the book):
#   #A Load pretrained tokenizer
#   #B Load pretrained model
#   #C Add semantic ID tokens to tokenizer
#   #D Resize embedding layer for new vocabulary
