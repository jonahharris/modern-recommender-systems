# Figure - Listing 5.9: Transformer cross-encoder reranker
# Source: chapters/ch05.md lines 616-650
# Chapter: 5
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
from transformers import AutoModel, AutoTokenizer
class TransformerCrossEncoder(nn.Module):
  def __init__(
    self,
    model_name: str = "bert-base-uncased",
    hidden_dim: int = 128,
  ):
    super().__init__()
    self.encoder = AutoModel.from_pretrained(model_name)  #A
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder_dim = self.encoder.config.hidden_size
    self.scorer = nn.Sequential(  #B
      nn.Linear(encoder_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, 1),
    )

  def forward(
    self,
    user_texts: list[str],
    item_texts: list[str],
  ) -> torch.Tensor:
    inputs = self.tokenizer(
      user_texts,
      item_texts,
      padding=True,
      truncation=True,
      max_length=512,
      return_tensors="pt",
    ).to(self.encoder.device)  #C

    outputs = self.encoder(**inputs)  #D
    cls_embedding = outputs.last_hidden_state[:, 0, :]  #E

    return self.scorer(cls_embedding).squeeze(-1)  #F

# Callout annotations (from the book):
#   #A Load a pretrained transformer as the encoder
#   #B A small scoring head on top of the encoder
#   #C Tokenize user history and item description as a text pair
#   #D Encode both inputs jointly with full self-attention
#   #E Use the \[CLS\] token representation as the combined embedding
#   #F Produce a relevance score
