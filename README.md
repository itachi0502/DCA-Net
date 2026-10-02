# DCA-Net

Research code associated with **DCA-Net: Graph-based dependent sampling and dynamic context association for biomedical trigger detection**.

Zilin Wang, Jiancheng Lv, and Xianggen Liu. *Information Sciences* **742** (2026), 123027. [Paper](https://doi.org/10.1016/j.ins.2025.123027).

This code-only release contains the available training, prediction, context-attention, CRF/focal-loss, and offline augmentation implementations. Datasets, generated text, pretrained weights, trained checkpoints, logs, and the published PDF are not included.

## Release scope

This is a curated snapshot of the research working directory, not a newly reproduced benchmark release. The source uses the historical `TAE` name for its context-attention module. The included `src/simple/tae_layer.py` implements multi-head attention with residual/feed-forward layers; it does not explicitly implement the sigmoid gate in Equation (11) of the paper. Therefore this snapshot should not be treated as a verified, exact implementation of every published equation or result.

The supplied working configuration had context attention disabled. `config/without_context.yaml` preserves that switch as an ablation example; `config/config.yaml` enables it. These are example configurations, not verified per-dataset settings for the paper's tables. Core model mathematics and sampling algorithms have not been rewritten for this release.

## Code map

| File | Purpose |
| --- | --- |
| `MCMC-LLAMA.py` | Context-aware MCMC trigger-pair sampling and Llama-compatible generation |
| `trigger_data_augment.py` | Alternative augmentation and sampling strategies |
| `src/simple/main.py` | Training, prediction, and evaluation entry point |
| `src/simple/model.py` | BioBERT, context attention, Transformer, and CRF composition |
| `src/simple/tae_layer.py` | Historical TAE context-attention implementation |
| `src/simple/crf_layer.py` | CRF decoding and focal-loss combination |
| `src/simple/train.py` | Optimizer, training loop, validation, and checkpoints |
| `src/simple/data_processor_s.py` | BIO/CoNLL loading and label mapping |
| `src/simple/predictor.py` | Checkpoint loading and prediction |
| `utils/metrics_new.py` | Label-based precision, recall, and F1 reporting |
| `convert_bio.py`, `covert_bio_fenci.py` | Original annotation-to-BIO conversion utilities |
| `BIO_Check*.py` | BIO label checking utilities |

Unused KB-NER retrieval code, duplicated experimental entry points, figure scripts, and third-party vendored packages were omitted. The original working directory was based on KB-NER; its bundled Flair MIT notice is retained under `THIRD_PARTY_LICENSES/`. That notice is not a new blanket license grant for this project's original contributions.

## Setup

Run commands from the repository root. A Python 3.8+ environment and a suitable PyTorch installation are needed; use a CUDA-compatible PyTorch build for GPU training.

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` is a minimal import-based dependency list, not a fully locked reproduction environment. `transformers==4.28.1` follows the original environment list. The optional spaCy converter additionally requires `spacy` and its `en_core_web_sm` model.

Obtain a BioBERT checkpoint and tokenizer separately, then set `paths.biobert_path` in the training YAML and `biobert_model` in augmentation YAML to the local directory or a compatible Hugging Face model identifier.

## Data format

Use two columns, `token label`, separated by whitespace, with a blank line between sentences. Labels use BIO notation. Prepare your authorized MLEE or BioNLP datasets separately and set the train/validation/test paths in YAML.

```text
Protein O
expression B-Gene_expression
increased B-Positive_regulation
. O
```

This is a synthetic format illustration, not benchmark data. Use the official splits; augment the training split only. The augmentation scripts copy validation/test files to their output directory without generating new examples for those splits. Check conversion choices and label inventories against the original dataset protocol before comparing results.

## Training and prediction

Edit `config/config.yaml`, especially dataset paths, BioBERT location, output directories, and `training.device` (`cuda:0` or `cpu`).

```bash
python -m src.simple.main --config config/config.yaml --mode train
python -m src.simple.main --mode predict \
  --model_path saved_models/bionlp11/dca_net/best_model \
  --input_file kb/datasets/bionlp11/test.txt \
  --output_file predictions/test.txt
python -m src.simple.main --mode evaluate \
  --model_path saved_models/bionlp11/dca_net/best_model \
  --test_file kb/datasets/bionlp11/test.txt
```

The default paths are examples. Update them for MLEE, BioNLP2011, or BioNLP2013 as appropriate. The included metric helper reports label-based metrics; do not assume equivalence to an official event/span scorer without checking the evaluation protocol.

## Offline augmentation

For the MCMC/Llama route, edit `mcmc_llama_config.yaml`, choose a provider-supported model, and export credentials locally:

```bash
export LLAMA_API_KEY='your-key'
export LLAMA_API_URL='https://your-provider/v1/chat/completions'
python MCMC-LLAMA.py --config mcmc_llama_config.yaml
```

Point the training configuration to the resulting `augmented_train.txt` if using augmented data. The API request sends sampled training sentences to the configured provider.

For the alternative augmentation script, set `LLM_API_KEY` and `LLM_BASE_URL` and edit `augmentation_config.yaml`:

```bash
python trigger_data_augment.py --config augmentation_config.yaml
```

Provider-specific model names must be checked against your service. `.env.example` documents variable names; the scripts read exported environment variables and do not automatically load `.env` files. Never commit credentials. No API calls or paid generation were performed while preparing this release.

## Publication preparation changes

- Removed embedded API credentials and private provider defaults; added environment-variable configuration.
- Replaced server-specific absolute paths with relative example paths.
- Corrected package imports in the main entry point and its evaluation mode.
- Added package markers, minimal dependencies, examples, and `.gitignore` exclusions.
- Kept the original attention-disabled switch in a separate ablation configuration.

See `VALIDATION.md` for checks and their limits. Training results from the paper have not been rerun as part of this code upload.

## Citation

```bibtex
@article{wang2026dcanet,
  title = {DCA-Net: Graph-based dependent sampling and dynamic context association for biomedical trigger detection},
  author = {Wang, Zilin and Lv, Jiancheng and Liu, Xianggen},
  journal = {Information Sciences},
  volume = {742},
  pages = {123027},
  year = {2026},
  doi = {10.1016/j.ins.2025.123027}
}
```
