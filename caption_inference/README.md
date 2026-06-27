# Caption inference

This directory contains the caption-generation utilities used by the UIT_HighDefinition ImageCLEFmedical Caption 2026 system.

## Files

| File | Purpose |
|---|---|
| `prompt_utils.py` | Builds the soft CUI-guided Qwen3-VL prompt. |
| `cleaner.py` | Applies conservative rule-based caption cleaning. |
| `merge_and_validate_submission.py` | Merges six rank outputs and validates the final submission. |
| `fix_submission_header.py` | Converts an internal lowercase `caption` header to the official `Caption` header. |
| `validate_final_submission.py` | Performs lightweight internal validation of a final `ID,Caption` CSV. |
| `notebooks/` | Sanitized Kaggle notebooks for the three-machine inference workflow. |

The notebooks use placeholders instead of private Kaggle dataset paths. See [`../PATH_CONFIG.md`](../PATH_CONFIG.md).
