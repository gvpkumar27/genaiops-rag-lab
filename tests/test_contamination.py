from pathlib import Path
import sys

# Ensure tests can import project modules from this repo layout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.check_contamination import (  # noqa: E402
    _find_exact_duplicates,
    _find_near_duplicates,
)


def test_exact_duplicate_detected_across_splits():
    splits = {
        "dev": [{"question": "What is RAG?"}],
        "test": [{"question": "What is rag?"}],
    }
    issues = _find_exact_duplicates(splits)
    assert issues


def test_near_duplicate_detected_across_splits():
    splits = {
        "dev": [{"question": "How do resumable uploads work for large files?"}],
        "test": [{"question": "How do upload resumes work for large files?"}],
    }
    issues = _find_near_duplicates(splits, threshold=0.5)
    assert issues
