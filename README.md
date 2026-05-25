# ettin-68m-nemotron-pii-onnx

Evaluation and publishing tooling for an ONNX export of [`kalyan-ks/ettin-68m-nemotron-pii`](https://huggingface.co/kalyan-ks/ettin-68m-nemotron-pii) — a 68M-parameter ModernBERT-family encoder fine-tuned by [Kalyan KS](https://huggingface.co/kalyan-ks) for PII detection across 55 entity types.

The published ONNX artifact lives at [`rulesentry-io/ettin-68m-nemotron-pii-onnx`](https://huggingface.co/rulesentry-io/ettin-68m-nemotron-pii-onnx). This repository contains the harness used to verify that export against its PyTorch parent and the workflow for re-publishing it.

## What's here

| File | Purpose |
|---|---|
| [`evaluate_onnx.py`](./evaluate_onnx.py) | Side-by-side evaluation harness — runs the ONNX export and the PyTorch parent on the Nemotron-PII test split and reports per-entity F1, precision, recall, throughput, and per-entity deltas. |
| [`gpu_smoke.py`](./gpu_smoke.py) | Post-publish verification that the HF artifact loads and runs on GPU. Three guard-asserts catch the common failure modes. |
| [`MODEL_CARD.md`](./MODEL_CARD.md) | The HuggingFace model card for the published artifact (staged into `onnx_fp32/README.md` at publish time). |
| [`publish.md`](./publish.md) | Step-by-step walkthrough for publishing the local ONNX export to HuggingFace. |
| [`onnx_fp32/`](./onnx_fp32) | Source artifact for the publish: `config.json` plus tokenizer files with hand-curated patches. `model.onnx` is gitignored — regenerate via `optimum-cli`. |
| [`CLAUDE.md`](./CLAUDE.md) | Architectural notes and known sharp edges in the harness. |

## Quick start

Environment: Python 3.12, [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv pip install "optimum[onnxruntime]" transformers torch datasets tqdm
uv run python evaluate_onnx.py
```

For GPU support, install `optimum[onnxruntime-gpu]` instead of `optimum[onnxruntime]`.

To regenerate the ONNX model file (gitignored) from the upstream PyTorch model:

```bash
uv run optimum-cli export onnx \
    --model kalyan-ks/ettin-68m-nemotron-pii \
    --task token-classification \
    ./onnx_fp32/
```

The tokenizer files in `onnx_fp32/` need to be present for the export to be usable end-to-end — `optimum-cli` doesn't bundle them automatically because of an upstream tokenizer-class loading bug. See [`CLAUDE.md`](./CLAUDE.md#known-issues) for the workaround.

## License

MIT. See [`LICENSE`](./LICENSE).

## Maintainers

[Mycos Technologies, Co., Ltd.](https://www.mycostech.com), as part of the [RuleSentry](https://www.rulesentry.io) platform.
