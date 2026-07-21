# Figure — Listing 7.18: Generating and decoding recommendations for a user
# Source: chapters/ch09.md lines 722-741
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Separate lookup for displaying history (UUID → title)
#   #B Show the user's last four watched items
#   #C Generate three recommendations
#   #D Decode each semantic ID to a title
formatter = BaseFormatter(item_df)
decoder = SemanticDecoder(item_df)
id_to_title = dict(zip(item_df['id'], item_df['title']))

user_inx = 6
user = training_data.iloc[user_inx]

print(f"User {user_inx} - Recent history:")
for i, raw_id in enumerate(user['history'][-4:]):
  title = id_to_title.get(raw_id, "Unknown")
  print(f"  {i+1}. {title}")

recs = generate_recommendations(
  model, tokenizer, formatter, user, num_items=3
)

print("\n--- Recommendations ---")
for i, raw_id in enumerate(recs):
  title = decoder.decode(raw_id)
  print(f"  {i+1}. {title} ({raw_id})")
