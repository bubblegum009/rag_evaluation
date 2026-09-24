"""
Text cleaning utilities for the RAG Experiment Lab.

This module removes common PDF extraction noise while preserving
meaningful text structure such as paragraph and section boundaries.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean extracted document text.

    The cleaning is intentionally conservative. It removes
    excessive whitespace without destroying meaningful
    paragraph boundaries.

    Args:
        text: Raw text extracted from a document.

    Returns:
        Cleaned text.
    """

    # Remove spaces at the beginning/end of each line.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines while preserving paragraph breaks.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace.
    text = text.strip()

    return text