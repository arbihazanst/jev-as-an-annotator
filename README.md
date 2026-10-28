# Decision Models Benchmark — Sentiment & Emotion

A reproducible benchmark package for comparing System-One / decision models on the same two classification datasets.

## Models currently wired in

- `nimble:9b` — SI-Lab native System-One endpoint
- `tev1:0.8b` — SI-Lab native System-One endpoint
- `tev1:4b` — SI-Lab native System-One endpoint
- `openai/gpt-6-luna-decisions` (default), `typesafe/jev-1.13`, or any OpenRouter decision model — OpenRouter Decisions API

Laya is **not** included yet because no Laya runner/API implementation was provided in the source files. It can be added later as long as it emits the same prediction schema and uses the same `question_for()` definitions.

## Scientific comparability

All models use the exact same benchmark definitions from `si_lab_decision_client.py` through `evaluate_decision_models.question_for()`:

### Sentiment

- `positive`
- `negative`
- `neutral`

Instruction: `Apa sentimen utama yang diungkapkan dalam teks ini?`

### Emotion

- `anger`
- `fear`
- `happy`
- `love`
- `sadness`

Instruction: `Apa satu emosi utama yang paling dominan dalam teks ini?`

The label descriptions/criteria are also shared unchanged. Do not create separate model-specific prompts for the final benchmark.

## Folder layout

```text
decision_models_benchmark_best_practices/
├── data/
│   ├── sentiments_data.csv
│   ├── emotionals_data.csv
│   └── turkiye_sentiments_data.csv
├── openrouter_only/
│   ├── __init__.py
│   ├── openrouter_common.py
│   ├── run_openrouter_smoke.py
│   └── run_openrouter_full.py
├── jev_only/
│   ├── __init__.py
│   ├── jev_common.py
│   ├── run_jev_smoke.py
│   └── run_jev_full.py
├── tests/
│   ├── test_best_practices.py
│   ├── test_evaluate_decision_models.py
│   └── test_si_lab_decision_client.py
├── results/
│   └── .gitkeep
├── dataset_registry.py
├── evaluate_decision_models.py
├── si_lab_decision_client.py
├── example_si_lab_decision.py
├── compare_all_models.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Key best-practice changes

1. **One task definition for every model.** Jev imports `question_for()` from the same evaluator used by Nimble/Tev1.
2. **Dry-run cannot contaminate real results.** Jev dry-runs validate data/payloads but do not create `predictions.jsonl`.
3. **Smoke and full outputs are isolated.** Default directories are timestamped and separated.
4. **Jev response validation is strict.** Missing/invalid labels or incomplete/non-finite probability vectors become errors rather than valid predictions.
5. **Full Jev runs are resumable.** Successful `(model, dataset, id)` records are skipped only when `--resume` is used on the exact output directory.
6. **Dataset hashes are stored.** Run manifests record SHA-256 hashes for reproducibility.
7. **No API keys in outputs.** Use environment variables or a local `.env` file.
8. **Final comparison is explicit.** `compare_all_models.py` combines prediction files and runs the same metrics implementation for every model.
9. **Coverage is checked before pairwise tests.** By default, every compared model must have the same successful IDs for each dataset. For the final paper add `--require-full-dataset`.
10. **Source dataset integrity is checked.** Aggregation verifies IDs, gold labels, and text hashes against the included CSVs.

## 1. Setup

Create a virtual environment and install dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add the credentials you actually use.

## 2. Run offline tests first

```bash
python -m unittest discover -s tests -v
```

Expected in the packaged version: **17 tests pass** without any API call.

## 3. Selecting Datasets (`--datasets`)

Every runner and comparison script accepts `--datasets <name> ...` to select one or more datasets.

### Available registered datasets:

| Dataset Name | Description | Language | Classes | File |
|---|---|---|---|---|
| `sentiment` | Indonesian 3-class sentiment | Indonesian | `positive`, `negative`, `neutral` | `data/sentiments_data.csv` |
| `emotion` | Indonesian 5-class emotion | Indonesian | `anger`, `fear`, `happy`, `love`, `sadness` | `data/emotionals_data.csv` |
| `sentiment_tr` | Turkish 3-class sentiment | Turkish | `positive`, `negative`, `neutral` | `data/turkiye_sentiments_data.csv` |
| `sentiment_hau` | Hausa 3-class sentiment | Hausa | `positive`, `negative`, `neutral` | `data/hausa_sentiments_data.csv` |

If `--datasets` is omitted, the runners default to all available registered datasets.

---

### Quick Start: Running ONLY the Türkiye Dataset (`sentiment_tr`)

```bash
# A. SI-Lab Dry-run (offline, validates dataset & questions without API calls)
python evaluate_decision_models.py \
  --dry-run \
  --datasets sentiment_tr \
  --samples-per-class 1 \
  --output results/si_lab_dryrun_tr

# B. SI-Lab Full Benchmark (Nimble + Tev1 via native System-One)
python evaluate_decision_models.py \
  --models nimble:9b tev1:0.8b tev1:4b \
  --datasets sentiment_tr \
  --transport systemone \
  --output results/si_lab_turkiye

# C. OpenRouter Smoke Test (default: openai/gpt-6-luna-decisions, or any decision model via --model)
python openrouter_only/run_openrouter_smoke.py \
  --model openai/gpt-6-luna-decisions \
  --datasets sentiment_tr \
  --samples-per-class 5 \
  --output results/openrouter_smoke_turkiye

# D. OpenRouter Full Benchmark
python openrouter_only/run_openrouter_full.py \
  --model openai/gpt-6-luna-decisions \
  --datasets sentiment_tr \
  --output results/openrouter_full_turkiye \
  --concurrency 5

# E. Final All-Model Comparison
python compare_all_models.py \
  --inputs results/si_lab_turkiye/predictions.jsonl results/openrouter_full_turkiye/predictions.jsonl \
  --models nimble:9b tev1:0.8b tev1:4b openai/gpt-6-luna-decisions \
  --datasets sentiment_tr \
  --require-full-dataset \
  --output results/all_models_turkiye
```

---

## 4. Validate SI-Lab pipeline without API calls (Dry-Run)

Validate all datasets:
```bash
python evaluate_decision_models.py \
  --dry-run \
  --samples-per-class 1 \
  --output results/si_lab_dryrun
```

Or validate a specific dataset (e.g. `sentiment_tr`):
```bash
python evaluate_decision_models.py \
  --dry-run \
  --datasets sentiment_tr \
  --samples-per-class 1 \
  --output results/si_lab_dryrun_tr
```

This validates datasets and writes reproducibility metadata, but sends no model requests.

## 5. SI-Lab full benchmark — Nimble + Tev1

By default, the runner connects directly to Ollama at `http://127.0.0.1:11435` with no API key required. (If using remote Open WebUI, configure `SI_LAB_API_KEY`).

Run on all datasets:
```bash
python evaluate_decision_models.py \
  --models nimble:9b tev1:0.8b tev1:4b \
  --datasets sentiment emotion sentiment_tr \
  --transport systemone \
  --output results/si_lab_full
```

Or run on a specific dataset:
```bash
python evaluate_decision_models.py \
  --models nimble:9b tev1:0.8b tev1:4b \
  --datasets sentiment_tr \
  --transport systemone \
  --output results/si_lab_turkiye
```

To resume after interruption, append `--resume`:
```bash
python evaluate_decision_models.py \
  --models nimble:9b tev1:0.8b tev1:4b \
  --datasets sentiment_tr \
  --transport systemone \
  --output results/si_lab_turkiye \
  --resume
```

Do not switch to the chat fallback for the final probability/calibration benchmark; native System-One is needed for probability vectors.

## 6. OpenRouter decision model smoke test

First validate locally with no OpenRouter call (dry-run):
```bash
python openrouter_only/run_openrouter_smoke.py \
  --model openai/gpt-6-luna-decisions \
  --dry-run \
  --datasets sentiment_tr \
  --samples-per-class 1
```

Run real smoke test on a specific dataset (e.g. `sentiment_tr`, 5 examples per class = 15 requests):
```bash
python openrouter_only/run_openrouter_smoke.py \
  --model openai/gpt-6-luna-decisions \
  --datasets sentiment_tr \
  --samples-per-class 5 \
  --output results/openrouter_smoke_gpt6_luna
```

Inspect:
- `results/openrouter_smoke_gpt6_luna/predictions.jsonl`
- `results/openrouter_smoke_gpt6_luna/summary.csv`
- actual OpenRouter `usage` values stored per row

## 7. OpenRouter full benchmark

Validate full payload construction without spending credits:
```bash
python openrouter_only/run_openrouter_full.py \
  --model openai/gpt-6-luna-decisions \
  --dry-run \
  --datasets sentiment_tr \
  --output results/openrouter_full_dryrun_gpt6_luna
```

Real full run on specific dataset:
```bash
python openrouter_only/run_openrouter_full.py \
  --model openai/gpt-6-luna-decisions \
  --datasets sentiment_tr \
  --output results/openrouter_full_gpt6_luna \
  --concurrency 5
```

Resume safely:
```bash
python openrouter_only/run_openrouter_full.py \
  --model openai/gpt-6-luna-decisions \
  --datasets sentiment_tr \
  --output results/openrouter_full_gpt6_luna \
  --concurrency 5 \
  --resume
```

Note: Any decision model available on OpenRouter can be benchmarked simply by changing `--model <model_name>` (e.g. `--model openai/gpt-6-luna-decisions` or `--model typesafe/jev-1.13`), or by setting `OPENROUTER_DECISION_MODEL` in your `.env`. The backward-compatible `jev_only/` scripts remain available as drop-in aliases.

If rate limits are frequent, reduce `--concurrency` rather than increasing it.

## 8. Final all-model comparison

After SI-Lab and Jev full runs complete successfully:
```bash
python compare_all_models.py \
  --inputs results/si_lab_turkiye/predictions.jsonl results/jev_full_turkiye/predictions.jsonl \
  --models nimble:9b tev1:0.8b tev1:4b typesafe/jev-1.13 \
  --datasets sentiment_tr \
  --require-full-dataset \
  --output results/all_models_turkiye
```

`--require-full-dataset` is recommended for the paper. The command will stop instead of silently producing pairwise statistics if any model is missing successful predictions for any dataset row.

## 9. Adding New Datasets (`dataset_registry.py`)

To add a new dataset in any language or schema:
1. Open `dataset_registry.py`.
2. Add an entry to `DATASET_REGISTRY`:
```python
DATASET_REGISTRY["my_dataset"] = DatasetSpec(
    name="my_dataset",
    path=DATA_DIR / "my_data.csv",
    text_column="text",
    label_column="label",
    # Set auto_labels=True to auto-detect labels from the CSV:
    auto_labels=True,
    answer_name="emotion",  # or "sentiment"
)
```
- **Auto-detected labels**: If `auto_labels=True` or `labels=()`, the labels are automatically read from the CSV.
- **Dynamic question generation**: Emotion questions dynamically build choice criteria matching whatever labels exist in the dataset (with built-in criteria for standard & extended emotions, and automatic fallbacks for custom labels).
- Once added, `--datasets my_dataset` works immediately across all benchmark and comparison scripts.

## 10. Final comparison artifacts

`results/all_models_final/` contains:

- `combined_predictions.jsonl`
- `comparison_manifest.json`
- `summary.csv`
- `per_class_metrics.csv`
- `calibration_bins.csv`
- `selective_coverage.csv`
- `pairwise_comparisons.csv`
- `confusion_matrices/`
- `visualizations/`
- `metrics_metadata.json`

Metrics produced by the shared evaluator include:

- Accuracy
- Balanced accuracy
- Macro-F1
- Weighted-F1
- 95% stratified bootstrap confidence intervals
- Per-class precision / recall / F1
- Confusion matrices
- Log loss
- Multiclass Brier score
- Expected Calibration Error (ECE)
- Selective coverage analysis
- Continuity-corrected pairwise McNemar tests

## 11. Prediction schema

Both SI-Lab and Jev runners write compatible records:

```json
{
  "model": "typesafe/jev-1.13",
  "dataset": "sentiment",
  "id": "123",
  "gold_label": "positive",
  "predicted_label": "positive",
  "probabilities": {
    "positive": 0.8,
    "negative": 0.1,
    "neutral": 0.1
  },
  "systemone_confidence": 0.8,
  "latency_ms": 123.4,
  "attempts": 1,
  "usage": {},
  "status": "ok",
  "error": null
}
```

## 12. Recommended experiment order

```text
1. python -m unittest discover -s tests -v
2. SI-Lab dry-run (--datasets sentiment_tr)
3. Jev dry-run (--datasets sentiment_tr)
4. Jev smoke test (--datasets sentiment_tr)
5. Check actual OpenRouter token usage/errors
6. SI-Lab full run (--datasets sentiment_tr)
7. Jev full run (--datasets sentiment_tr)
8. Verify zero unresolved errors / complete coverage
9. compare_all_models.py --datasets sentiment_tr --require-full-dataset
10. Freeze final prediction files + manifests for the paper
```

## Important note on latency/cost comparisons

Accuracy/F1/calibration are directly comparable because the task/data/labels are shared. Latency and cost should be reported with context because the models are served through different infrastructure (SI-Lab/Ollama vs OpenRouter/TypeSafe), so service/network overhead is part of the observed latency and not purely model inference time.