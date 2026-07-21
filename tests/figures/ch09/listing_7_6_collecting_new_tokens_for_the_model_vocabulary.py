# Figure — Listing 7.6: Collecting new tokens for the model vocabulary
# Source: chapters/ch09.md lines 209-214
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Initialize empty token set
#   #B Scan all users
#   #C Collect unique tokens
def find_new_tokens(formatter, train_data):
  new_tokens = set()
  for user_inx in range(train_data.shape[0]):
    prompt = formatter.format(train_data.iloc[user_inx], is_training=False)
    new_tokens.update(prompt.split())
  return new_tokens
