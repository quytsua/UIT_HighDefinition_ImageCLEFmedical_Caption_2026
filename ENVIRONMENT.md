# Environment

The caption inference pipeline was executed on Kaggle GPU runtimes. Machine 1, Machine 2, and Machine 3 used the same Kaggle GPU runtime recipe, the same model, the same dependency installation pattern, the same 4-bit NF4 quantization configuration, and the same deterministic greedy decoding setup.

The values below are taken from the Machine 1 execution log, which corresponds to a successful inference run.

## Runtime

| Component | Value |
|---|---|
| Python executable | `/usr/bin/python3` |
| Python runtime | Python 3.12 environment |
| GPU | Tesla T4 |
| CUDA available | `True` |
| Visible GPU count | `1` |
| Model device | `cuda:0` |

## Core packages

| Package | Version |
|---|---:|
| `numpy` | `2.0.2` |
| `pandas` | `2.2.2` |
| `Pillow` | `11.3.0` |
| `scipy` | `1.15.3` |
| `scikit-learn` | `1.6.1` |
| `torch` | `2.10.0+cu128` |
| `transformers` | `5.8.0.dev0` |
| `huggingface_hub` | `1.13.0` |
| `qwen-vl-utils` | `0.0.14` |
| `accelerate` | `>=1.2.0` |
| `bitsandbytes` | `>=0.43.3` |
| `safetensors` | `>=0.4.5` |

The `transformers` package was installed directly from the Hugging Face GitHub repository during the Kaggle run. Because this corresponds to a development build, exact reproduction may require using the same installation recipe rather than relying on a fixed PyPI release.

## Inference configuration

| Item | Value |
|---|---|
| Vision-language model | `Qwen/Qwen3-VL-8B-Instruct` |
| Quantization | 4-bit NF4 with `bitsandbytes` |
| Compute dtype | `float16` |
| Double quantization | Enabled |
| Decoding | Deterministic greedy decoding |
| Sampling | `do_sample=False` |
| Maximum new tokens | `80` |
| Image pixel range | `min_pixels = 256 * 28 * 28`, `max_pixels = 512 * 28 * 28` |
| Distributed inference | Six-rank modulo sharding |

## Notes

The public repository does not include private execution logs. The values above are summarized from the successful execution logs and are provided for reproducibility documentation.
