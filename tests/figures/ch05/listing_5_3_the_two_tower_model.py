# Figure — Listing 5.3: The two-tower model
# Source: chapters/ch05.md lines 121-154
# Chapter: 5
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Two separate embedding layers, one per tower
#   #B Each tower is a small feedforward network
#   #C Sigmoid maps the dot product to a probability
#   #D Binary cross-entropy loss for the positive/negative classification task
#   #E Regularization strength
#   #F Pass the first item through its embedding layer and tower
#   #G Pass the second item through its embedding layer and tower
#   #H Element-wise product, sum, and sigmoid produce the affinity score
#   #I Binary cross-entropy between prediction and label
#   #J L2 regularization prevents embedding weights from growing too large
class TwoTower(nn.Module):
  def __init__(self, num_items, emb_dim, c_vector=1e-6):
    super().__init__()
    self.embedding1 = nn.Embedding(num_items, emb_dim)
    self.embedding2 = nn.Embedding(num_items, emb_dim)
    self.tower_one = nn.Sequential(
      nn.Linear(emb_dim, 128),
      nn.ReLU(),
      nn.Linear(128, emb_dim)
    )
    self.tower_two = nn.Sequential(
      nn.Linear(emb_dim, 128),
      nn.ReLU(),
      nn.Linear(128, emb_dim)
    )
    self.sig = nn.Sigmoid()
    self.bce = nn.BCEWithLogitsLoss()
    self.c_vector = c_vector

  def forward(self, item_1, item_2):
    emb1 = self.tower_one(self.embedding1(item_1))
    emb2 = self.tower_two(self.embedding2(item_2))
    score = self.sig(
      torch.sum(emb1 * emb2, dim=1, dtype=torch.float)
    )
    return score

  def loss(self, pred, label):
    bce_loss = self.bce(pred, label)
    reg = sum(
      torch.sum(p ** 2.0) for p in
      [self.embedding1.weight, self.embedding2.weight]
    ) * self.c_vector
    return bce_loss + reg
