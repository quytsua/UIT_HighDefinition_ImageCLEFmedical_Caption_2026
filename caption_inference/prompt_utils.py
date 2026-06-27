"""Prompt construction utilities for soft CUI-guided caption generation."""

PROMPT_TEMPLATE = """You are generating a caption for a medical figure in a biomedical article.

Task:
Write one concise, clinically relevant English caption for the image.

Rules:
- Focus on the visible medical content.
- Mention modality when clear: CT, MRI, X-ray, ultrasound, histology, endoscopy, angiography, PET/CT, microscopy.
- Mention anatomical region and main visual finding if visible.
- Do not write "This image shows".
- Do not write "Caption:".
- Do not include uncertainty unless needed.
- Do not invent patient age, sex, diagnosis, treatment, or outcome if not visible.
- Output only one caption sentence.

{hint}
Final caption:
""".strip()


def build_qwen3_prompt(cui_terms: str = "") -> str:
    """Build the final text prompt for Qwen3-VL."""
    hint = ""
    if cui_terms and len(str(cui_terms).strip()) > 0:
        hint = f"""Possible UMLS/CUI clinical hints: {cui_terms}
Use these hints only if they are visually supported by the image.
Do not force unsupported diseases or findings.
"""
    return PROMPT_TEMPLATE.format(hint=hint)
