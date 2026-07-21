## Base Architecture: Res8

### Context
A CNN architecture was needed for keyword spotting, small enough to
train quickly, but grounded in established research rather than an
ad-hoc design.

### Decision
**Res8** (Tang and Lin, 2018) was implemented from scratch in
PyTorch, consisting of an input convolutional layer, followed by 3
residual blocks with 2 convolutions and batch normalization each,
average pooling, and a final fully connected layer.

### Alternatives considered
A plain VGG-style CNN was considered, but it is more prone to
vanishing gradients without skip connections and is less grounded in
KWS literature. Res8-narrow was considered as well, but since it is
already a compressed variant, using it as a starting point would
pre-empt the compression experiments this project is meant to
explore. Res15, with 6 residual blocks, was also considered, but it
is slower to train and not necessary for this project's scope.

### Consequences
- Positive: Grounded in a well-known KWS baseline with published
  results to compare against, around 94 percent accuracy on 12
  classes in the original paper.
- Positive: The residual connections give the compression study an
  additional angle, since pruning skip connections is non-trivial and
  worth discussing separately from pruning plain convolutional
  layers.
- Negative: Implemented independently and not verified numerically
  against the original TensorFlow implementation.