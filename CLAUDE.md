# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Two Python scripts:

- **`evaluate_onnx.py`** — fidelity evaluation harness that benchmarks the ONNX export against its PyTorch parent on the `nvidia/Nemotron-PII` test split. Used to verify the published ONNX artifact matches the PyTorch source.
- **`gpu_smoke.py`** — post-publish verification that the published artifact loads and runs on GPU (requires `optimum[onnxruntime-gpu]`). Three guard-asserts catch the common failure modes: CUDA provider not registered, model loaded but silently fell back to CPU, inference produces wrong output. Called manually after publishing.

Supporting docs:

- **`README.md`** — GitHub-facing README for the repository.
- **`MODEL_CARD.md`** — HuggingFace model-card prose. Staged as `README.md` inside `./onnx_fp32/` at publish time (HF requires the filename `README.md` to render as the model card).

For ModernBERT-family architecture details (RoPE / GeGLU / sliding-window / no-bias settings), reference the upstream Ettin paper (arxiv.org/abs/2507.11412) or clone the JHU-CLSP training repo (`github.com/JHU-CLSP/ettin-encoder-vs-decoder`) — its `pretraining/configs/standard/*/encoder_*.yaml` files contain the exact hyperparameters.

There is no application code, no tests, no CI.

## Local artifacts (not in HF Hub)

- **`./onnx_fp32/`** — fp32 ONNX export of `kalyan-ks/ettin-68m-nemotron-pii`. Source of the published artifact at `rulesentry-io/ettin-68m-nemotron-pii-onnx`. Contains `model.onnx` (gitignored — regenerate via the `optimum-cli` command in Commands below), tokenizer files (with `model_input_names` patched to `["input_ids", "attention_mask"]`), and `config.json`.

## Commands

Environment is a `uv`-managed venv (`.venv/`, Python 3.12). All commands assume `uv` is on PATH.

```bash
# Install deps (CPU)
uv pip install "optimum[onnxruntime]" transformers torch datasets tqdm

# GPU also (optional — script auto-detects CUDA and adds GPU configs)
uv pip install "optimum[onnxruntime-gpu]"

# Run the evaluation against the local fp32 export
uv run python evaluate_onnx.py

# GPU smoke test of the *published* HF artifact (requires onnxruntime-gpu)
uv run python gpu_smoke.py
```

Top-of-file knobs in `evaluate_onnx.py`: `TEST_SAMPLES` (default 10_000; set `None` for full 50k split, or 128 for a smoke run), `BATCH_SIZE`, `FP32_ONNX_DIR`.

The script pulls the PyTorch model + dataset from HF Hub at runtime; first run will be slow and requires network. All ONNX inputs come from local `./onnx_fp32/`.

**Regenerating `./onnx_fp32/`** (if missing):

```bash
uv run optimum-cli export onnx \
    --model kalyan-ks/ettin-68m-nemotron-pii \
    --task token-classification \
    ./onnx_fp32/
```

`optimum-cli` won't bundle the tokenizer because the upstream tokenizer config triggers the `TokenizersBackend does not exist` bug (Known Issue #3). After export, tokenizer files must be loaded from a working source, saved with `tokenizer.save_pretrained('./onnx_fp32/')`, then the saved `tokenizer_config.json` must be patched to set `model_input_names = ["input_ids", "attention_mask"]` (the runtime `model_input_names` override on the tokenizer object does *not* serialize).

## Architecture notes for `evaluate_onnx.py`

Two things in this script are non-obvious and easy to break:

1. **Tokenizer source for the PyTorch path** in `build_pipeline`. The PyTorch branch loads its tokenizer from `./onnx_fp32/`, *not* from the PyTorch model's HF repo. Two reasons: (a) the upstream tokenizer at `kalyan-ks/ettin-68m-nemotron-pii` fails to load (`TokenizersBackend does not exist`); (b) `./onnx_fp32/`'s `tokenizer_config.json` has `model_input_names` already patched to `["input_ids", "attention_mask"]`, which is what `ModernBertForTokenClassification.forward()` accepts (it rejects `token_type_ids`). If you ever change `FP32_ONNX_DIR` or rebuild it, re-verify the tokenizer files are present and `model_input_names` is correctly patched.

2. **`preds_to_set` post-processing** is load-bearing for quality numbers. The HF token-classification pipeline with `aggregation_strategy="simple"` (a) keeps leading whitespace attached to a span and (b) fails to merge contiguous same-label spans (e.g. a date `1963-08-08` comes back as 5 separate spans). The function sorts by start offset, merges adjacent same-label spans (strict touch — no gap tolerance), then trims edge whitespace using the original text. Gold spans are exact `(start, end, label)` tuples; any drift in this normalization shows up directly as F1 loss. Don't "simplify" this without re-running both configs.

Everything else is straightforward: `evaluate_config` builds a pipeline per (kind, device) tuple, `run_inference` times pipeline calls only (not tokenization-only or model-only), `compute_metrics` does exact-match span PRF overall and per entity, and the three `print_*` helpers format the final tables. The per-entity delta table compares ONNX-fp32 vs PyTorch-CPU and is expected to show ~0 across all classes when the export is faithful.

## Model context

Upstream/base model lineage: `kalyan-ks/ettin-68m-nemotron-pii` (fine-tune) ← `jhu-clsp/ettin-encoder-68m` (base, 68M ModernBERT-family encoder) ← trained on `nvidia/Nemotron-PII`. All MIT.

## Known issues we resolved (FYI when reading `evaluate_onnx.py`)

These are sharp edges that already bit us once. The fixes are in code; this list exists so a future reader doesn't accidentally undo them.

1. **`datasets` 4.8.5 row iteration crash** — iterating the dataset with `for s in dataset` triggered `TypeError: unhashable type: 'set'` inside `Features.decode_example`. Fix: read columns directly (`dataset["text"]`, `dataset["spans"]`), which goes through `decode_column` instead.
2. **`spans` is a stringified list, not a Sequence[dict]** — upstream dataset stores spans as the Python `repr` of a list (single-quoted), declared as `Value('string')` in the schema. Must be parsed with `ast.literal_eval`, not `json.loads`. Handled in `spans_to_set`.
3. **`TokenizersBackend does not exist`** — `AutoTokenizer.from_pretrained("kalyan-ks/ettin-68m-nemotron-pii")` fails because the upstream tokenizer config references an unimportable class. Fix: load the tokenizer from `./onnx_fp32/` (its tokenizer files were vendored from a working source and have a loadable tokenizer class, with `model_input_names` already patched to omit `token_type_ids`).
4. **`token_type_ids` rejected by `ModernBertForTokenClassification.forward()`** — see Architecture note 1 above. The `./onnx_fp32/tokenizer_config.json` has the patched `model_input_names` already, so loading the tokenizer from that directory avoids this.
5. **Catastrophic F1 from naive scoring** — `aggregation_strategy="simple"` (a) attaches leading whitespace to spans (start offset off by 1) and (b) splits multi-token entities like dates into many tiny spans. Without the merge + trim in `preds_to_set`, both ONNX and PyTorch produce single-digit F1. See Architecture note 2 above.
6. **`save_pretrained()` doesn't serialize runtime `model_input_names` overrides** — setting `tokenizer.model_input_names = [...]` then calling `tokenizer.save_pretrained(dir)` does NOT persist the override into `tokenizer_config.json` (it's a class-level attribute, not a config field). Fix: directly write `model_input_names` into `tokenizer_config.json` after `save_pretrained`. Relevant when rebuilding `./onnx_fp32/`.
