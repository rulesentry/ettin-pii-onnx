---
license: mit
language:
  - en
tags:
  - onnx
  - pii
  - pii-detection
  - ner
  - PII
  - privacy
  - token-classification
  - modernbert
library_name: transformers
base_model: kalyan-ks/ettin-68m-nemotron-pii
base_model_relation: finetune
datasets:
  - nvidia/Nemotron-PII
pipeline_tag: token-classification
metrics:
  - f1
  - precision
  - recall
---

# ettin-68m-nemotron-pii-onnx

**fp32 ONNX export of [`kalyan-ks/ettin-68m-nemotron-pii`](https://huggingface.co/kalyan-ks/ettin-68m-nemotron-pii) — bit-for-bit faithful to the PyTorch parent | 68M Parameters | 55 PII Entity Types**

This is an ONNX export of the `ettin-68m-nemotron-pii` model, onnx export produced by [Mycos Technologies, Co., Ltd.](https://www.mycostech.com) for use in [RuleSentry](https://www.rulesentry.io) and other high-throughput, CPU-bound PII detection pipelines. The underlying model detects 55 PII entity types across healthcare, finance, legal, and cybersecurity domains.

---

## Overview

`ettin-68m-nemotron-pii` is a ModernBERT-based encoder fine-tuned on NVIDIA's synthetic [Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) dataset. This is an **fp32 (unquantized) ONNX export** of that model — bit-for-bit faithful to the PyTorch parent on the Nemotron-PII test split (see [Evaluation](#evaluation)) and modestly faster on CPU via ONNX Runtime graph optimizations.

**Why ONNX?**
- Modest CPU speedup over PyTorch via ONNX Runtime graph optimizations
- No PyTorch dependency at runtime
- Native integration with non-Python runtimes (Rust, C#, Java, Go, etc.)
- Smaller runtime footprint — ONNX Runtime is ~25 MB on disk vs PyTorch's ~700 MB

The graph takes two inputs (`input_ids`, `attention_mask`) and exposes one output (`logits`). It was exported with `optimum-cli` using the ModernBERT-specific ONNX configuration, so `token_type_ids` are omitted from both the graph and the bundled tokenizer config.

---

## Usage

### With Optimum + ONNX Runtime (recommended)

```python
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer, pipeline

model_id = "rulesentry-io/ettin-68m-nemotron-pii-onnx"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForTokenClassification.from_pretrained(model_id)

ner = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=-1,  # pin to CPU; see "Using GPU" below for CUDA setup
)

text = "Jane Doe's SSN is 123-45-6789 and her email is jane.doe@example.com"
entities = ner(text)
print(entities)
# [{'entity_group': 'first_name', ...}, {'entity_group': 'ssn', ...}, {'entity_group': 'email', ...}]
```

### With ONNX Runtime directly

```python
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

model_id = "rulesentry-io/ettin-68m-nemotron-pii-onnx"
tokenizer = AutoTokenizer.from_pretrained(model_id)

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4  # tune for your CPU
session = ort.InferenceSession("model.onnx", sess_options=sess_options,
                               providers=["CPUExecutionProvider"])

text = "John Smith lives at 42 Elm Street, Boston."
inputs = tokenizer(text, return_tensors="np")
outputs = session.run(None, dict(inputs))
logits = outputs[0]
```

### Using GPU

For CUDA-accelerated inference, install the GPU build of ONNX Runtime instead of the default:

```bash
pip install "optimum[onnxruntime-gpu]" transformers
```

Then pass `device=0` to the pipeline:

```python
ner = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0,
)
```

Or set the provider list on a direct `InferenceSession`:

```python
session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

The trailing `CPUExecutionProvider` lets ORT degrade gracefully if CUDA initialization fails at runtime.

**Why isn't GPU the default?** `transformers.pipeline` auto-detects `torch.cuda.is_available()` and tries to use a CUDA execution provider. If you have CUDA-enabled PyTorch installed but only the CPU build of ONNX Runtime (`optimum[onnxruntime]`), this fails with `Asked to use CUDAExecutionProvider … not available`. The `device=-1` in the recommended example is a defensive default that works regardless of your environment.

> **Install (CPU):** `pip install "optimum[onnxruntime]" transformers`

---

## Evaluation

### Fidelity to the PyTorch parent

This ONNX export was evaluated head-to-head against the upstream PyTorch model on a 10,000-sample slice of the `nvidia/Nemotron-PII` test split. Predictions are effectively identical:

- **54 of 55 entity classes** show **zero** F1 delta between ONNX-fp32 and PyTorch (CPU). Predictions match span-for-span.
- **1 entity class** (`mac_address`) shows a +0.09 F1 difference — within numerical-noise floor from operator-implementation differences between PyTorch and ONNX Runtime kernels.
- **No quality regressions** on any entity class.

This means all per-entity quality characteristics of the parent model carry over to this export without modification. For absolute F1 / Precision / Recall numbers (overall and per-entity), refer to the upstream model card: [`kalyan-ks/ettin-68m-nemotron-pii`](https://huggingface.co/kalyan-ks/ettin-68m-nemotron-pii). Note that span-level NER scores depend significantly on the choice of scorer (token-level vs strict-span vs span-overlap merging); choose the convention that matches your downstream use case.

### Performance

Same 10,000-sample slice, CPU, batch size 32:

| Runtime | Throughput | Latency (batch p50) | Relative speed |
|---|---:|---:|---:|
| PyTorch (CPU) | 21.9 samples/s | 1246 ms | 1.00× |
| **ONNX-fp32 (CPU)** | **24.5 samples/s** | **1077 ms** | **1.12×** |

Throughput is hardware-dependent. The speedup comes from ONNX Runtime's graph optimizations (operator fusion, constant folding, kernel selection) — not from quantization. Numbers will vary with CPU architecture, thread count, and batch size.

---

## Supported PII Entity Types

This model detects **55 PII entity types** across structured and unstructured text:

| Entity | Description |
|---|---|
| `account_number` | Account Number |
| `age` | Age |
| `api_key` | API Key |
| `bank_routing_number` | Bank Routing Number |
| `biometric_identifier` | Biometric Identifier |
| `blood_type` | Blood Type |
| `certificate_license_number` | Certificate or License Number |
| `city` | City |
| `company_name` | Company Name |
| `coordinate` | Geographic Coordinate |
| `country` | Country |
| `county` | County |
| `credit_debit_card` | Credit or Debit Card Number |
| `customer_id` | Customer ID |
| `cvv` | Card Verification Value (CVV) |
| `date` | Date |
| `date_of_birth` | Date of Birth |
| `date_time` | Date and Time |
| `device_identifier` | Device Identifier |
| `education_level` | Education Level |
| `email` | Email Address |
| `employee_id` | Employee ID |
| `employment_status` | Employment Status |
| `fax_number` | Fax Number |
| `first_name` | First Name |
| `gender` | Gender |
| `health_plan_beneficiary_number` | Health Plan Beneficiary Number |
| `http_cookie` | HTTP Cookie |
| `ipv4` | IPv4 Address |
| `ipv6` | IPv6 Address |
| `language` | Language |
| `last_name` | Last Name |
| `license_plate` | Vehicle License Plate |
| `mac_address` | MAC Address |
| `medical_record_number` | Medical Record Number |
| `national_id` | National Identification Number |
| `occupation` | Occupation |
| `password` | Password |
| `phone_number` | Phone Number |
| `pin` | Personal Identification Number (PIN) |
| `political_view` | Political View |
| `postcode` | Postcode / Zip Code |
| `race_ethnicity` | Race or Ethnicity |
| `religious_belief` | Religious Belief |
| `sexuality` | Sexuality / Sexual Orientation |
| `ssn` | Social Security Number |
| `state` | State |
| `street_address` | Street Address |
| `swift_bic` | SWIFT / BIC Code |
| `tax_id` | Tax Identification Number |
| `time` | Time |
| `unique_id` | Unique Identifier |
| `url` | URL / Web Address |
| `user_name` | Username |
| `vehicle_identifier` | Vehicle Identification Number (VIN) |

---

## Model Lineage & Credits

This repository contains an ONNX export produced by [RuleSentry.IO](https://huggingface.co/rulesentry-io). The original model and all training were done upstream:

| Component | Source | Description |
|---|---|---|
| Fine-tuned PII model | [`kalyan-ks/ettin-68m-nemotron-pii`](https://huggingface.co/kalyan-ks/ettin-68m-nemotron-pii) by [@kalyan-ks](https://huggingface.co/kalyan-ks) | ModernBERT fine-tuned for PII NER |
| Base encoder | [`jhu-clsp/ettin-encoder-68m`](https://huggingface.co/jhu-clsp/ettin-encoder-68m) by Johns Hopkins CLSP | 68M parameter ModernBERT encoder |
| Training dataset | [`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII) by NVIDIA | Synthetic PII dataset, 55 entity types |

All upstream components are licensed under **MIT**.

---

## Limitations

All quality characteristics of this export are inherited from the upstream PyTorch model (see [Evaluation](#evaluation) for the fidelity proof). Limitations below are upstream-inherited:

- **English only** — the model is trained on English-language text and performs poorly on other languages.
- **Occupation entity** — `occupation` has a known low F1 score in upstream evaluations and should be treated with caution.
- **Synthetic training data** — trained on NVIDIA's synthetic Nemotron-PII dataset; real-world distributions (especially niche domains) may yield lower performance. Evaluate on representative samples of your own data before deploying.
- **Context length** — very long documents should be chunked before inference.
- **Not a legal compliance tool** — PII detection is probabilistic. Do not use as a sole control for regulatory compliance (GDPR, HIPAA, CCPA) without human review.

For applications where structured identifiers (`ipv4`, `mac_address`, `credit_debit_card`, `ssn`, etc.) must be detected with very high recall, consider supplementing this model with deterministic regex/format validation — these entity types have well-defined formats that pattern matching handles more reliably than any neural model.

---

## About

This model is maintained by **[Mycos Technologies, Co., Ltd.](https://www.mycostech.com)**, a Thai software company building privacy and data governance tooling. It is part of the [RuleSentry](https://www.rulesentry.io) platform, which provides automated PII detection and data compliance infrastructure.

- 🌐 Product: [rulesentry.io](https://www.rulesentry.io)
- 🤗 HuggingFace org: [rulesentry-io](https://huggingface.co/rulesentry-io)
- 🛠️ Evaluation tooling: [github.com/rulesentry/ettin-pii-onnx](https://github.com/rulesentry/ettin-pii-onnx)

---

## Citation

If you use this model, please cite the upstream fine-tuned model and dataset:

```bibtex
@misc{ettin-68m-pii-2026,
  title     = {ettin-68m-nemotron-pii-2026: PII Detection Model},
  author    = {Kalyan KS},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/kalyan-ks/ettin-68m-nemotron-pii}
}

@misc{nvidia2025nemotronpii,
  title  = {Nemotron-PII: Synthesized Data for Privacy-Preserving AI},
  author = {NVIDIA},
  year   = {2025},
  url    = {https://huggingface.co/datasets/nvidia/Nemotron-PII}
}
```

This ONNX export:

```bibtex
@misc{rulesentry2026ettin-onnx,
  title     = {ettin-68m-nemotron-pii-onnx: ONNX Export for CPU Inference},
  author    = {RuleSentry.IO},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/rulesentry-io/ettin-68m-nemotron-pii-onnx}
}
```

