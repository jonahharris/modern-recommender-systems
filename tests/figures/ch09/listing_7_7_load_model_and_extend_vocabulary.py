# Figure — Listing 7.7: Load model and extend vocabulary
# Source: chapters/ch09.md lines 236-241
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Load pretrained tokenizer
#   #B Load pretrained model
#   #C Add semantic ID tokens to tokenizer
#   #D Resize embedding layer for new vocabulary
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.add_tokens(list(new_tokens))
model.resize_token_embeddings(len(tokenizer))
