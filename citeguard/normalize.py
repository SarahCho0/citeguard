"""Text normalization utilities.

Absorbs harmless surface variation ("same sentence, different notation"):
full-width vs half-width characters, smart-quote styles, whitespace runs,
letter case. When an LLM transcribes a quote, these variations are noise;
everything after this preprocessing layer is a deterministic string
comparison.
"""

from __future__ import annotations

import re
import unicodedata

# Unify smart quotes and full-width punctuation into plain ASCII
_QUOTE_MAP = str.maketrans({
    "“": '"', "”": '"',
    "‘": "'", "’": "'",
    "–": "-", "—": "-",
    "·": " ",                 # middle dot is treated as whitespace
})

_WS = re.compile(r"\s+")
_ZERO_WIDTH = re.compile(r"[​‌‍﻿]")


def normalize(text: str) -> str:
    """Convert text to its canonical comparison form.

    NFKC normalization (full-width → half-width etc.) → quote unification
    → zero-width character removal → lowercasing → whitespace-run collapse
    → edge trim.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTE_MAP)
    text = _ZERO_WIDTH.sub("", text)
    text = text.lower()
    text = _WS.sub(" ", text)
    return text.strip()


def squash(text: str) -> str:
    """Canonical form with all whitespace removed.

    Used for char n-gram matching of text with unreliable spacing —
    e.g. Korean search queries, where spacing is highly inconsistent.
    """
    return normalize(text).replace(" ", "")
