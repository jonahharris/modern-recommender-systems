# Figure — Listing 7.8: Configure the training pipeline
# Source: chapters/ch09.md lines 259-298
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Custom dataset replacing deprecated TextDataset
#   #B Keep only full-length blocks
#   #C Labels equal input\_ids for causal LM
#   #D Formatted user histories, one per line
#   #E Maximum sequence length in tokens
#   #F Causal LM mode (predict next token)
#   #G Number of passes over the dataset
#   #H Higher LR for new vocabulary
#   #I Checkpoint interval
class SemanticIDDataset(Dataset):
  def __init__(self, tokenizer, file_path, block_size=128):
    with open(file_path, 'r') as f:
      text = f.read()
    tokenized = tokenizer(
      text, truncation=True, max_length=block_size,
      return_overflowing_tokens=True, return_length=True,
    )
    self.examples = [
      torch.tensor(ids)
      for ids, length in zip(tokenized['input_ids'],
                             tokenized['length'])
      if length == block_size
    ]

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx):
    return {"input_ids": self.examples[idx],
            "labels": self.examples[idx]}

train_dataset = SemanticIDDataset(
  tokenizer=tokenizer,
  file_path="train_recsys.txt",
  block_size=128
)

data_collator = DataCollatorForLanguageModeling(
  tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
  output_dir="./recsys_gpt",
  overwrite_output_dir=True,
  num_train_epochs=2,
  per_device_train_batch_size=2,
  learning_rate=5e-4,
  save_steps=1000
)
