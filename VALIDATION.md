# Release validation

Checked on 2026-10-02:

- All 20 Python files parse successfully.
- All three primary command-line entry points load and display help: `python -m src.simple.main --help`, `python MCMC-LLAMA.py --help`, and `python trigger_data_augment.py --help`.
- A CPU smoke check used a tiny, randomly initialized local BERT (hidden size 16), a synthetic BIO sentence, context attention, Transformer, CRF, and focal loss. Dataset loading, finite loss, backward propagation, and prediction decoding passed. No pretrained weights or external API were required for this check.
- Runtime checks used the existing research environment with PyTorch 1.13.1+cu117 and Transformers 4.28.1. No shared environment was modified.
- Release files were checked against the API key strings found in the source, common credential patterns, and the original private absolute path prefixes. No matches remained.
- The release includes no benchmark datasets, generated corpora, trained/pretrained weights, logs, or installer binaries.

These checks establish basic packaging and component execution only. They do not establish paper-result reproduction, full training convergence, provider compatibility, or equivalence to official benchmark scorers. The paper/code differences described in README remain unresolved in the source snapshot.
