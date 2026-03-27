# Synthetic Augmentation Policy

## Scope

Synthetic data is only used when required source fields are unavailable from APIs.
The ratio of fully synthetic rows must remain less than or equal to 30% of the final dataset.

## Justification

- Several APIs expose accuracy and timing but omit GPU metadata or cost.
- Academic comparison requires a complete schema across all rows.

## Controls

1. Preserve all direct API values unchanged.
2. Add provenance metadata in future commits for inferred fields.
3. Keep deterministic generation using a fixed random seed.
4. Log synthetic ratio and assumptions in `reports/data_manifest.json`.

## Status

Synthetic generation is implemented with deterministic sampling, hard cap checks, and manifest logging.
