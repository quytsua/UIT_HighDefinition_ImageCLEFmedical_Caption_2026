"""Rule-based caption cleaner used in the UIT_HighDefinition captioning pipeline."""

import re
import pandas as pd

BAD_PREFIXES = [
    r"^caption\s*:\s*",
    r"^answer\s*:\s*",
    r"^output\s*:\s*",
    r"^final caption\s*:\s*",
    r"^this image shows\s+",
    r"^the image shows\s+",
    r"^this figure shows\s+",
    r"^the figure shows\s+",
    r"^an image of\s+",
    r"^a medical image of\s+",
]

GENERIC_BAD = {
    "the relevant anatomical region",
    "relevant anatomical region",
    "medical image showing findings",
    "the image shows the relevant anatomy",
    "an image of the relevant anatomical region",
    "medical imaging figure",
    "medical image",
}

MODALITY_REPLACEMENTS = [
    (r"\bX-Ray Computed Tomography\b", "CT"),
    (r"\bComputed Tomography\b", "CT"),
    (r"\bcomputed tomography\b", "CT"),
    (r"\bComputerized Tomography\b", "CT"),
    (r"\bMagnetic Resonance Imaging\b", "MRI"),
    (r"\bmagnetic resonance imaging\b", "MRI"),
    (r"\bUltrasonography\b", "ultrasound"),
    (r"\bultrasonography\b", "ultrasound"),
    (r"\bSonography\b", "ultrasound"),
    (r"\bsonography\b", "ultrasound"),
    (r"\bPlain x-ray\b", "X-ray"),
    (r"\bplain x-ray\b", "X-ray"),
    (r"\bplain radiograph\b", "X-ray"),
]

TRUNCATED_ENDINGS = [
    r"\blikely\.$",
    r"\bsuggestive of\.$",
    r"\bindicative of\.$",
    r"\bconsistent with\.$",
    r"\bwith no\.$",
    r"\blocated in the\.$",
    r"\bat the\.$",
    r"\bdemonstrating\.$",
    r"\bshowing\.$",
]


def clean_caption_v4(text: str, max_words: int = 55) -> str:
    """Clean one generated medical image caption.

    This function performs conservative surface-level normalization only.
    It does not verify visual evidence or clinical factual correctness.
    """
    if pd.isna(text):
        return ""

    s = str(text).strip()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip("`").strip()
    s = s.strip("\"'“”‘’").strip()

    for pat in BAD_PREFIXES:
        s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()

    s = re.sub(r"(?i)^write one concise.*?caption[:\s-]*", "", s).strip()
    s = re.sub(r"(?i)^generate one concise.*?caption[:\s-]*", "", s).strip()
    s = re.sub(r"(?i)^final caption[:\s-]*", "", s).strip()

    for pat, rep in MODALITY_REPLACEMENTS:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)

    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()

    words = s.split()
    if len(words) > max_words:
        sentences = re.split(r"(?<=[.!?])\s+", s)
        if len(sentences) > 1:
            s = sentences[0].strip()
            if len(s.split()) < 8 and len(sentences) > 1:
                s = (s + " " + sentences[1]).strip()
        else:
            s = " ".join(words[:max_words]).strip()

    for pat in TRUNCATED_ENDINGS:
        if re.search(pat, s, flags=re.IGNORECASE):
            s = re.sub(pat, ".", s, flags=re.IGNORECASE).strip()

    s = s.strip()

    if s.lower().strip(". ") in GENERIC_BAD:
        return ""

    if s and s[-1] not in ".!?":
        s += "."

    return s
