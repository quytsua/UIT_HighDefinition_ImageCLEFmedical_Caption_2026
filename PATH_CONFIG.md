# Path configuration guide

The public notebooks use placeholders instead of private Kaggle paths. Before running the notebooks, replace the placeholders below with your own authorized Kaggle dataset or notebook-output paths.

## Caption inference placeholders

| Placeholder | Meaning | Example target |
|---|---|---|
| `<TEST_IMAGE_DATASET>` | Kaggle dataset containing the official test images | `/kaggle/input/your-test-dataset/TEST DATASET/images` |
| `<CUI_SUBMISSION_DATASET>` | Kaggle dataset containing auxiliary CUI predictions | `/kaggle/input/your-cui-submission/submission.csv` |
| `<CUI_METADATA_DATASET>` | Kaggle dataset/notebook output containing `cui2meta.json` | `/kaggle/input/your-cui-metadata/cui2meta.json` |
| `<PREVIOUS_RANK_OUTPUTS>` | Previous rank CSV outputs for Machine 1 resume | `/kaggle/input/your-rank-backup/qwen3_full_6gpu/...` |
| `<MACHINE2_PREVIOUS_OUTPUTS>` | Previous rank CSV outputs for Machine 2 resume | `/kaggle/input/your-machine2-output/qwen3_full_6gpu/...` |
| `<MACHINE3_PREVIOUS_OUTPUTS>` | Previous rank CSV outputs for Machine 3 resume | `/kaggle/input/your-machine3-output/qwen3_full_6gpu/...` |
| `<FINAL_SUBMISSION_NOTEBOOK_OUTPUT>` | Notebook output containing the merged final submission | `/kaggle/input/your-final-submission-output/...` |

## Notes

- Do not commit private Kaggle dataset slugs, tokens, API credentials, or restricted datasets.
- The ImageCLEFmedical datasets are not redistributed in this repository.
- `/kaggle/working` paths are generic Kaggle output paths and may appear in notebooks/scripts.
