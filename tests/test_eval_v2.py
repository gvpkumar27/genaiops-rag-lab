from pathlib import Path
import sys

# Ensure tests can import project modules from this repo layout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval_v2 import (  # noqa: E402
    _citation_metrics,
    _groundedness,
    _keyword_coverage,
    _lexical_f1,
    _unknown_behavior,
)


def test_keyword_coverage():
    assert _keyword_coverage("Java stream api uses collections", ["stream", "collections"]) == 1.0


def test_lexical_f1_with_reference():
    val = _lexical_f1("default methods improve compatibility", "default methods provide backward compatibility")
    assert val is not None
    assert 0.0 < val <= 1.0


def test_citation_metrics_support_string_and_object_gold():
    citations = [{"source": "data/docs/Java.pdf", "chunk_id": 3}]
    gold = ["data/docs/Java.pdf|3", {"source": "data/docs/Java.pdf", "chunk_id": 5}]
    recall, precision = _citation_metrics(citations, gold)
    assert recall == 0.5
    assert precision == 1.0


def test_groundedness():
    answer = "default methods backward compatibility"
    cites = [{"text_preview": "Default methods provide backward compatibility for interfaces."}]
    val = _groundedness(answer, cites)
    assert val > 0.5


def test_unknown_behavior():
    assert _unknown_behavior("I don't know based on the provided documents.", True) == 1.0
    assert _unknown_behavior("Some answer", True) == 0.0
    assert _unknown_behavior("I don't know based on the provided documents.", False) == 0.0
