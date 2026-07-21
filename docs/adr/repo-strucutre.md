## Python and C++ Separation via a Binary File

### Context
A decision was needed on how the Python training code and the C++
compression and inference code would communicate with each other.

### Decision
The two parts are fully separated. Python exports trained weights to
a custom binary format, and C++ reads that file independently, with
no bindings such as pybind11 connecting the two.

### Alternatives considered
Using pybind11 was considered, since it would allow calling C++ code
directly from Python, but it adds build complexity that is not
relevant to this project's goal.

### Consequences
- Positive: Mirrors real embedded deployment workflows, where a model
  is trained in Python and then deployed as static data inside C or
  C++ firmware.
- Positive: Results in a simpler CMake build, without dependencies on
  Python headers or interpreter versions.
- Negative: There is no convenient way to call the C++ code from
  Python for quick testing, so verification has to happen directly on
  the C++ side.

## Repository Layout by Language

### Context
A folder structure was needed that reflects the strict Python and C++
separation described above.

### Decision
The repository is split at the top level by language, into `python/`
and `cpp/`, with a shared `format/weights_format.md` file documenting
the binary contract between them.

### Alternatives considered
Splitting by pipeline stage, into folders such as `training/`,
`compression/`, and `verification/`, was considered, but it would mix
languages within each stage, which does not fit naturally with the
strict file-based separation chosen above.

### Consequences
- Positive: Each side, `python/` and `cpp/`, is a self-contained and
  independently buildable project.
- Positive: The weight format is documented in one explicit place,
  instead of living only in code on both sides.

## Checkpoints Store state_dict, Not the Full Model Object

### Context
A way was needed to save and load trained weights that would survive
future changes to the model code.

### Decision
`model.state_dict()` is always saved, and the full model object is
never saved directly through `torch.save(model, path)`.

### Consequences
- Positive: Checkpoints are not tied to the exact class definition or
  import path that was used at save time.
- Negative: The model class has to be re-instantiated before the
  weights can be loaded.