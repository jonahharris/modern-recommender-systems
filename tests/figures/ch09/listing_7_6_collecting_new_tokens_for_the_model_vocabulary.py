# Figure - Listing 7.6: Collecting new tokens for the model vocabulary
# Source: chapters/ch09.md lines 209-214
# Chapter: 9
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def find_new_tokens(formatter, train_data):
  new_tokens = set()  #A
  for user_inx in range(train_data.shape[0]):  #B
    prompt = formatter.format(train_data.iloc[user_inx], is_training=False)
    new_tokens.update(prompt.split())  #C
  return new_tokens

# Callout annotations (from the book):
#   #A Initialize empty token set
#   #B Scan all users
#   #C Collect unique tokens
