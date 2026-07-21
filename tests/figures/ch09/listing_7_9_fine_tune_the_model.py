# Figure - Listing 7.9: Fine-tune the model
# Source: chapters/ch09.md lines 326-333
# Chapter: 9
# Category: needs-training  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
trainer = Trainer(
  model=model,
  args=training_args,
  data_collator=data_collator,
  train_dataset=train_dataset,
)  #A
trainer.train()  #B
print("Fine-tuning complete. The model now speaks 'RecSys'!")

# Callout annotations (from the book):
#   #A Instantiate trainer with all components
#   #B Start fine-tuning
