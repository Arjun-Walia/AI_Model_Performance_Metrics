# AI Performance Metrics Analysis and Optimization

This repository contains an academic, reproducible project for analyzing and optimizing AI model performance across accuracy, latency, throughput, memory, and compute cost.

## Project Structure

```text
/project-root
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|-- src/
|   |-- common/
|   `-- collection/
|-- reports/
|-- config/
|-- README.md
`-- requirements.txt
```

## Current Status

Phase 1 implementation currently includes collection + merge pipeline:

- Loads and validates the bootstrap dataset from `data/raw/ai_model_experiments_5000.csv`
- Fetches best-effort records from PapersWithCode, OpenML, and Hugging Face APIs
- Normalizes API records into the required schema and deduplicates merged rows
- Applies deterministic synthetic augmentation only when needed, capped by `synthetic_max_ratio`
- Enforces schema and metric range checks
- Writes curated output to `data/processed/phase1_bootstrap_curated.csv`
- Writes run manifest to `reports/data_manifest.json`

If an API endpoint is unavailable, the pipeline skips that source and continues with available data.
Synthetic rows (if generated) are tagged with provenance columns: `source_name`, `is_synthetic`, `synthetic_fields`, `confidence_score`.
Manifest metadata includes reproducibility fields such as `random_seed`, `timezone`, and output CSV SHA-256 checksum.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Optional: copy `.env.example` to `.env` and set API tokens.

## Run Phase 1 Baseline

```bash
python -m src.cli --config config/config.yaml
```

Expected output:

- Curated CSV in `data/processed/`
- Manifest JSON in `reports/`

## Quality Check

Run tests plus a pipeline smoke check with one command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_quality.ps1
```

## Dataset Schema

The project preserves this required core schema:

`experiment_id, Model, dataset, gpu_type, batch_size, accuracy, latency_ms, tokens_per_second, memory_usage_gb, compute_cost_usd, run_timestamp`

Optional provenance columns are appended for traceability.
