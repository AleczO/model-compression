## Dataset Choice

### Context
Needed to select a dataset for training a small CNN, suitable for
demonstrating model compression (pruning/quantization) later in the
project.

### Decision
**Google Speech Commands v0.02** was chosen, since keyword spotting is
a canonical TinyML task and closely matches the kind of models
Antmicro works with.

### Alternatives considered
CIFAR-10 with a small CNN was considered, since it is simpler and
well-documented, but it is generic and less relevant to embedded ML
use cases. MedMNIST was also considered, since it fits a medical
imaging background, but it has a weaker connection to embedded and
TinyML applications.

### Consequences
- Positive: Directly relevant to embedded ML and keyword spotting use
  cases.
- Positive: Well-studied problem with existing baselines (Res8/Res15)
  to compare against.
- Negative: Requires an extra preprocessing step, since waveforms need
  to be converted to mel-spectrograms before training.

## Class Scope: 14 Commands, Unknown, and Background

### Context
Google Speech Commands v0.02 contains 35 words in total: 14 command
words (10 classic ones and 4 added in v0.02), plus 21 auxiliary words
that are intended for building an "unknown" category in the standard
KWS methodology.

### Decision
The 14 command words are used as separate classes, together with one
aggregated "unknown" class, undersampled from the 21 auxiliary words,
and one "background" class, built from 1 second clips randomly cut
from `_background_noise_`.

### Alternatives considered
Full 35-way classification was considered, since it is closer to a
general audio classification benchmark, but it represents a different
task than realistic keyword spotting.

### Consequences
- Positive: Matches realistic keyword-spotting framing and is aligned
  with the embedded and TinyML use case.
- Positive: Results in fewer output classes and a smaller final fully
  connected layer.
- Negative: Requires manual undersampling to balance "unknown" against
  the individual command classes.

## Train, Validation, and Test Split

### Context
A way was needed to split files into train, validation, and test sets
without leaking the same speaker's recordings across splits.

### Decision
The official `validation_list.txt` and `testing_list.txt` files are
used, filtered down to the chosen classes, instead of building a
custom random split.

### Alternatives considered
A custom random split was considered, since it would be simpler to
implement, but it risks leaking the same speaker into both train and
test, which would inflate accuracy.

### Consequences
- Positive: Produces a speaker-disjoint split, comparable to the
  results reported in the original dataset paper (Warden, 2018).
- Negative: Requires separate, manual handling for
  `_background_noise_`, since it is not covered by the official split
  lists.

## Mel-Spectrogram Instead of MFCC

### Context
The reference architecture (Tang and Lin, Res8/Res15) uses MFCCs as
input, which apply a DCT step on top of the mel filterbank.

### Decision
Log-mel spectrograms are used directly, and the DCT step is skipped.

### Alternatives considered
Using MFCCs was considered, in order to match the original paper's
input pipeline exactly.

### Consequences
- Positive: Results in a simpler preprocessing pipeline.
- Positive: Consistent with common practice in newer CNN-based KWS
  work, since CNNs do not require decorrelated input features in the
  way GMM/HMM systems historically did.
- Negative: Not a one-to-one reproduction of the original paper's
  input pipeline.