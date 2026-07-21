# Figure — Listing 7.16: Generating semantic ID recommendations
# Source: chapters/ch09.md lines 617-652
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Format user history using the same formatter used during training
#   #B Each item is four tokens (L1, L2, L3, LF)
#   #C Enable sampling for diverse recommendations
#   #D Isolate newly generated tokens from the prompt
#   #E Only collect valid semantic ID tokens
#   #F Leaf node signals end of one item
def generate_recommendations(model, tokenizer, formatter,
                             user_data, num_items=3):
  model.eval()
  device = model.device

  prompt_text = formatter.format(user_data, is_training=False)
  inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

  max_new_tokens = num_items * 4

  with torch.no_grad():
    output_ids = model.generate(
      inputs["input_ids"],
      max_new_tokens=max_new_tokens,
      pad_token_id=tokenizer.eos_token_id,
      do_sample=True,
      top_k=50,
      temperature=0.8
    )

  full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  generated_part = full_text[len(prompt_text):].strip()

  tokens = generated_part.split()
  items = []
  current = []
  for t in tokens:
    if t.startswith(("L1_", "L2_", "L3_", "LF_")):
      current.append(t)
      if t.startswith("LF_"):
        items.append(" ".join(current))
        current = []
        if len(items) >= num_items:
          break

  return items
