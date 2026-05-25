"""
Replication evaluation for rulesentry-io/ettin-68m-nemotron-pii-onnx
against its upstream PyTorch parent kalyan-ks/ettin-68m-nemotron-pii,
on the nvidia/Nemotron-PII test split.

Two questions answered:
  1. Quality — does the ONNX export preserve span-level F1?
     Measured by running both ONNX and PyTorch through the same pipeline
     on the same slice. Quality is hardware-independent; only the CPU pass
     is used for the quality comparison.
  2. Performance — latency / throughput per (model, device) config.
     ONNX-CPU and PyTorch-CPU always run; ONNX-GPU and PyTorch-GPU also
     run if CUDA is available (and onnxruntime-gpu is installed for ONNX-GPU).

Dataset schema:
  text  : str
  spans : str  (Python-repr of a list of {"start","end","label",...})

Install:
    uv pip install "optimum[onnxruntime]" transformers torch datasets tqdm
    # For GPU also:
    uv pip install "optimum[onnxruntime-gpu]"

Usage:
    uv run python evaluate_onnx.py
"""

import ast
import statistics
import time
from collections import defaultdict

from datasets import load_dataset
from optimum.onnxruntime import ORTModelForTokenClassification
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
FP32_ONNX_DIR     = "./onnx_fp32"           # local fp32 ONNX export
PYTORCH_MODEL_ID  = "kalyan-ks/ettin-68m-nemotron-pii"
DATASET_ID        = "nvidia/Nemotron-PII"
TEST_SPLIT        = "test"
TEST_SAMPLES      = 10_000    # set None for full 50k split, or 128 for a smoke run
BATCH_SIZE        = 32
# ─────────────────────────────────────────────────────────────────────────────


def build_pipeline(kind: str, device: int):
    """Build a token-classification pipeline.

    kind   : "onnx-fp32" or "pytorch"
    device : -1 for CPU, 0 for first CUDA device
    """
    provider = "CUDAExecutionProvider" if device >= 0 else "CPUExecutionProvider"

    if kind == "onnx-fp32":
        tokenizer = AutoTokenizer.from_pretrained(FP32_ONNX_DIR)
        model = ORTModelForTokenClassification.from_pretrained(
            FP32_ONNX_DIR,
            provider=provider,
        )
    elif kind == "pytorch":
        # Upstream tokenizer fails to load (TokenizersBackend bug); use the
        # bundled fp32 export's tokenizer files (same vocab, model_input_names
        # already trimmed to ['input_ids', 'attention_mask'] which is what
        # ModernBertForTokenClassification.forward() expects).
        tokenizer = AutoTokenizer.from_pretrained(FP32_ONNX_DIR)
        model = AutoModelForTokenClassification.from_pretrained(PYTORCH_MODEL_ID)
    else:
        raise ValueError(f"unknown kind: {kind}")

    return pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )


def load_test_data():
    print(f"Loading {DATASET_ID} [{TEST_SPLIT}] ...")
    ds = load_dataset(DATASET_ID, split=TEST_SPLIT)
    if TEST_SAMPLES:
        ds = ds.select(range(TEST_SAMPLES))
    print(f"  {len(ds)} samples loaded")
    return ds


def spans_to_set(spans):
    """Convert a list of span dicts (or its repr-string form) to a set of
    (start, end, label) tuples."""
    if isinstance(spans, str):
        spans = ast.literal_eval(spans)
    return {(s["start"], s["end"], s["label"]) for s in spans}


def preds_to_set(predictions, text):
    """Convert pipeline output to a set of (start, end, label) tuples.

    The token-classification pipeline with aggregation_strategy='simple'
    often (a) keeps the leading whitespace token attached to a span and
    (b) fails to merge contiguous same-label spans (e.g. a date like
    '1963-08-08' comes back as 5 separate spans). We post-process here:
      1. sort by start
      2. merge spans that share a label and are adjacent (end_i >= start_j)
      3. trim whitespace at the edges of the merged span using `text`
    """
    preds = sorted(predictions, key=lambda x: x["start"])
    merged = []
    for p in preds:
        label = p["entity_group"]
        start, end = p["start"], p["end"]
        if merged and merged[-1][2] == label and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, label])

    out = set()
    for s, e, label in merged:
        snippet = text[s:e]
        lstrip = len(snippet) - len(snippet.lstrip())
        rstrip = len(snippet) - len(snippet.rstrip())
        out.add((s + lstrip, e - rstrip, label))
    return out


def run_inference(ner, texts, batch_size):
    """Return (pred_spans, timing). Timing covers only the pipeline calls."""
    pred_spans = []
    batch_times = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        t0 = time.perf_counter()
        batch_pred = ner(batch)
        batch_times.append(time.perf_counter() - t0)
        for text, preds in zip(batch, batch_pred):
            pred_spans.append(preds_to_set(preds, text))
    total = sum(batch_times)
    return pred_spans, {
        "total_s":          total,
        "throughput_s_per": len(texts) / total if total > 0 else 0.0,
        "batch_ms_p50":     statistics.median(batch_times) * 1000,
        "batch_ms_mean":    statistics.mean(batch_times) * 1000,
    }


def compute_metrics(all_gold_sets, all_pred_sets):
    total_tp = total_fp = total_fn = 0
    entity_tp = defaultdict(int)
    entity_fp = defaultdict(int)
    entity_fn = defaultdict(int)

    for gold_set, pred_set in zip(all_gold_sets, all_pred_sets):
        tp_spans = gold_set & pred_set
        fp_spans = pred_set - gold_set
        fn_spans = gold_set - pred_set

        total_tp += len(tp_spans)
        total_fp += len(fp_spans)
        total_fn += len(fn_spans)

        for (_, _, label) in tp_spans:
            entity_tp[label] += 1
        for (_, _, label) in fp_spans:
            entity_fp[label] += 1
        for (_, _, label) in fn_spans:
            entity_fn[label] += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    overall = prf(total_tp, total_fp, total_fn)
    per_entity = {
        label: prf(entity_tp[label], entity_fp[label], entity_fn[label])
        for label in set(entity_tp) | set(entity_fp) | set(entity_fn)
    }
    return overall, per_entity


def evaluate_config(label, kind, device, texts, gold_spans):
    print(f"\n── {label} ──")
    try:
        ner = build_pipeline(kind, device)
    except Exception as e:
        print(f"  skipped: {type(e).__name__}: {e}")
        return None
    preds, timing = run_inference(ner, texts, BATCH_SIZE)
    overall, per_entity = compute_metrics(gold_spans, preds)
    return {
        "label":      label,
        "kind":       kind,
        "device":     device,
        "overall":    overall,
        "per_entity": per_entity,
        "timing":     timing,
    }


def print_quality_table(results):
    print("\n" + "=" * 70)
    print("QUALITY  (exact span match)")
    print("=" * 70)
    print(f"  {'Config':<22} {'F1':>8} {'Precision':>11} {'Recall':>9}")
    print("-" * 70)
    for r in results:
        p, rc, f = r["overall"]
        print(f"  {r['label']:<22} {f*100:>7.2f}  {p*100:>10.2f}  {rc*100:>8.2f}")


def print_perf_table(results):
    print("\n" + "=" * 70)
    print("PERFORMANCE")
    print("=" * 70)
    print(f"  {'Config':<22} {'Total (s)':>10} {'Throughput':>14} {'Batch p50':>12}")
    print("-" * 70)
    for r in results:
        t = r["timing"]
        print(
            f"  {r['label']:<22} {t['total_s']:>10.2f} "
            f"{t['throughput_s_per']:>9.1f} s/s  "
            f"{t['batch_ms_p50']:>9.1f} ms"
        )


def print_per_entity_delta(onnx_res, pyt_res):
    """Per-entity delta between ONNX-CPU and PyTorch-CPU.
    For fp32 ONNX this should be near-zero across all classes;
    any non-trivial delta indicates an export-fidelity issue."""
    if onnx_res is None or pyt_res is None:
        return
    labels = sorted(set(onnx_res["per_entity"]) | set(pyt_res["per_entity"]))
    print("\n" + "=" * 75)
    print("PER-ENTITY F1  (ONNX-fp32-CPU vs PyTorch-CPU)")
    print("=" * 75)
    print(f"  {'Entity':<34} {'PyT F1':>8} {'ONNX F1':>9} {'Δ':>8}")
    print("-" * 75)

    def f1(res, lab):
        return res["per_entity"].get(lab, (0.0, 0.0, 0.0))[2] * 100

    rows = [(lab, f1(pyt_res, lab), f1(onnx_res, lab)) for lab in labels]
    rows.sort(key=lambda r: r[2] - r[1])  # worst regressions first
    for lab, p_f1, o_f1 in rows:
        delta = o_f1 - p_f1
        print(f"  {lab:<34} {p_f1:>7.2f}  {o_f1:>8.2f}  {delta:>+7.2f}")


def run_evaluation():
    dataset = load_test_data()
    texts      = dataset["text"]
    gold_spans = [spans_to_set(s) for s in dataset["spans"]]

    print(f"\nCUDA available: {CUDA_AVAILABLE}")
    configs = [
        ("ONNX-fp32 (CPU)", "onnx-fp32", -1),
        ("PyTorch (CPU)",   "pytorch",   -1),
    ]
    if CUDA_AVAILABLE:
        configs += [
            ("ONNX-fp32 (GPU)", "onnx-fp32", 0),
            ("PyTorch (GPU)",   "pytorch",   0),
        ]
    else:
        print("(skipping GPU runs)")

    results = []
    for label, kind, device in configs:
        r = evaluate_config(label, kind, device, texts, gold_spans)
        if r is not None:
            results.append(r)

    print_quality_table(results)
    print_perf_table(results)

    onnx_cpu = next((r for r in results if r["label"] == "ONNX-fp32 (CPU)"), None)
    pyt_cpu  = next((r for r in results if r["label"] == "PyTorch (CPU)"), None)
    print_per_entity_delta(onnx_cpu, pyt_cpu)

    return results


if __name__ == "__main__":
    run_evaluation()
