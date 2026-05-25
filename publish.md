# Publishing `./onnx_fp32/` to HuggingFace

Walkthrough for publishing the local fp32 ONNX export to `rulesentry-io/ettin-68m-nemotron-pii-onnx`.

## What ships

```
rulesentry-io/ettin-68m-nemotron-pii-onnx/
├── README.md              ← from project root's MODEL_CARD.md (HF requires this filename)
├── model.onnx             ← from ./onnx_fp32/   (~268 MB, LFS-tracked automatically)
├── config.json            ← from ./onnx_fp32/
├── tokenizer.json         ← from ./onnx_fp32/   (~3.5 MB, also LFS)
├── tokenizer_config.json  ← from ./onnx_fp32/   (patched model_input_names)
└── special_tokens_map.json
```

That's 6 files. Nothing else. No `ort_config.json`, no quant artifacts, no eval scripts, no `CLAUDE.md`, no `publish.md`.

## Step 1: Get a write token

1. Visit https://huggingface.co/settings/tokens
2. Create a token with **write** scope (read scope can't upload). If you already have a write token saved, skip.
3. Copy it — looks like `hf_…`.

## Step 2: Authenticate locally

```bash
uv run hf auth login
# paste the token when prompted; "add to git credentials" can be declined — see notes
```

Token is stored in `~/.cache/huggingface/token`. Verify with:

```bash
uv run hf auth whoami
```

Should print your HF username and the orgs you belong to (look for `rulesentry-io`).

## Step 3: Check repo state

The repo was taken down — need to know how. Three possibilities:

```bash
# This curl tells you the repo's current state without auth needed
curl -sI https://huggingface.co/rulesentry-io/ettin-68m-nemotron-pii-onnx | head -1
```

- **200 OK** → repo exists and is public. Just push, files will overwrite.
- **401/403** → repo exists but is private. Need to be logged in as a member of `rulesentry-io` to access it. Verify with `hf auth whoami`. Push works the same as the public case; visibility can be toggled later in the repo's Settings tab on the web.
- **404** → repo deleted, need to recreate:
  ```bash
  uv run hf repo create rulesentry-io/ettin-68m-nemotron-pii-onnx --repo-type model
  ```

## Step 4: Stage the files

The repo root needs `README.md` alongside the model files. HuggingFace requires the file be named `README.md` to render as the model card on the page. Copy our `MODEL_CARD.md` into the upload directory under that name:

```bash
cp ./MODEL_CARD.md ./onnx_fp32/README.md
```

Now `./onnx_fp32/` contains exactly the 6 files that should land in the repo.

## Step 5: Verify upload contents

`hf upload` has no `--dry-run` flag, so verify manually before pushing. The contents of `./onnx_fp32/` are what gets uploaded — list them and confirm against the manifest at the top of this doc:

```bash
ls -la ./onnx_fp32/
```

Expect exactly:

- `README.md` (staged in Step 4)
- `model.onnx`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

If anything extra is in the directory (e.g., a stray `ort_config.json` from quantization experiments, a `.DS_Store`, etc.), delete it before upload — `hf upload` of a directory uploads everything in it.

Then confirm the README will render as intended:

```bash
head -30 ./onnx_fp32/README.md
```

Make sure the front-matter and title look right.

## Step 6: Upload

```bash
uv run hf upload \
    rulesentry-io/ettin-68m-nemotron-pii-onnx \
    ./onnx_fp32 \
    . \
    --repo-type model \
    --commit-message "fp32 ONNX export, bit-for-bit faithful to PyTorch parent"
```

Reads as: "upload local folder `./onnx_fp32` to remote path `.` (repo root) in repo `rulesentry-io/ettin-68m-nemotron-pii-onnx`, type=model."

Takes a few minutes — `model.onnx` is ~268 MB and gets uploaded over HF's LFS endpoint with multipart chunks. Per-file progress bars stream as it goes.

If your connection drops mid-upload, just re-run the same command. HF deduplicates by file hash and resumes.

## Step 7: Verify on the web

Visit https://huggingface.co/rulesentry-io/ettin-68m-nemotron-pii-onnx and check:

1. **README renders correctly** — the front-matter at the top (license, tags, base_model) populates the right-side sidebar with the model metadata. Tags and license should show up as chips.
2. **Files tab** shows the 6 files, with `model.onnx` and `tokenizer.json` marked as LFS-stored.
3. **Model card body** displays the fp32 framing, Evaluation section, etc. — no "experimental" banner, no quant caveats.

## Step 8: Smoke test the published artifact

End-to-end check from a clean state — make sure what's on the Hub actually works:

```bash
uv run python -c "
from optimum.onnxruntime import ORTModelForTokenClassification
from transformers import AutoTokenizer, pipeline

model_id = 'rulesentry-io/ettin-68m-nemotron-pii-onnx'
tok = AutoTokenizer.from_pretrained(model_id)
print('tokenizer model_input_names:', tok.model_input_names)

mdl = ORTModelForTokenClassification.from_pretrained(model_id)
# device=-1 pins to CPU. Without this, transformers.pipeline auto-detects
# torch.cuda.is_available() and tries to use CUDAExecutionProvider, which
# requires onnxruntime-gpu (not installed in the default deps).
ner = pipeline('token-classification', model=mdl, tokenizer=tok,
               aggregation_strategy='simple', device=-1)

text = \"Jane Doe's SSN is 123-45-6789 and her email is jane.doe@example.com\"
for e in ner(text):
    print(e)
"
```

Should print `['input_ids', 'attention_mask']` for the tokenizer (confirming the patched config made it to the Hub) and then the entities. If this works, the publish is done.

## Things to know

- **HF caches.** If you already downloaded the prior (taken-down) quantized version, `~/.cache/huggingface/hub/models--rulesentry-io--ettin-68m-nemotron-pii-onnx/` has stale files. Before the smoke test in step 8, either delete that cache directory or pass `force_download=True`, to make sure you're testing the fresh upload.

- **No tags / no releases.** HF doesn't have releases like GitHub. The `main` branch on the model repo is the only revision unless you create branches. For this case, plain `main` is fine.

- **Discoverability.** The `pipeline_tag: token-classification` and `tags` in the README front-matter are what make this show up in HF's filter UI. Already set correctly.

- **License field.** Set to MIT in the front-matter. Inherits from upstream Ettin + Nemotron, both MIT. Consistent.

- **Cleanup after.** The `./onnx_fp32/README.md` you staged in step 4 is just a copy of `MODEL_CARD.md`. Delete it after upload (`rm ./onnx_fp32/README.md`) so you don't accidentally push a stale copy on the next round.

**Last sanity check before pushing:** open `./onnx_fp32/README.md` and skim it. That's the first thing every visitor to the model page will see. If anything reads as stale, fix it before step 6.
