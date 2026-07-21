# Figure — Listing 7.9: Fine-tune the model
# Source: chapters/ch09.md lines 326-333
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Instantiate trainer with all components
#   #B Start fine-tuning
trainer = Trainer(
  model=model,
  args=training_args,
  data_collator=data_collator,
  train_dataset=train_dataset,
)
trainer.train()
print("Fine-tuning complete. The model now speaks 'RecSys'!")
