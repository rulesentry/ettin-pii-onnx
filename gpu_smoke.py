"""GPU smoke test for the published ONNX model.

Requires `optimum[onnxruntime-gpu]` (not the CPU-only `optimum[onnxruntime]`).
Run after publishing to verify the artifact loads and runs on GPU.

Usage:
    uv pip install "optimum[onnxruntime-gpu]"
    uv run python gpu_smoke.py
"""

import onnxruntime as ort
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer, pipeline

MODEL_ID = "rulesentry-io/ettin-68m-nemotron-pii-onnx"

# 1. Sanity check: is the CUDA provider even registered?
providers = ort.get_available_providers()
print("available providers:", providers)
assert "CUDAExecutionProvider" in providers, (
    "CUDAExecutionProvider not registered — install optimum[onnxruntime-gpu] "
    "and verify your CUDA toolkit matches the onnxruntime-gpu build."
)

# 2. Load the model with an explicit CUDA provider. The explicit argument
# makes the test fail loudly if ORT can't actually use GPU, instead of
# silently falling back to CPU.
tok = AutoTokenizer.from_pretrained(MODEL_ID)
print("tokenizer model_input_names:", tok.model_input_names)

mdl = ORTModelForTokenClassification.from_pretrained(
    MODEL_ID,
    provider="CUDAExecutionProvider",
)
print("model providers           :", mdl.providers)
assert "CUDAExecutionProvider" in mdl.providers, (
    "Model loaded without CUDA — ORT silently fell back to CPU. "
    "Usually a CUDA toolkit version mismatch with the onnxruntime-gpu wheel."
)

# 3. Run inference. device=0 routes the pipeline through CUDA.
ner = pipeline(
    "token-classification",
    model=mdl,
    tokenizer=tok,
    aggregation_strategy="simple",
    device=0,
)

text = "Jane Doe's SSN is 123-45-6789 and her email is jane.doe@example.com"
for e in ner(text):
    print(e)
