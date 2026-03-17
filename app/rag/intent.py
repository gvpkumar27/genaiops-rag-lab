import re

SUMMARY = "summary"
DEFINITION = "definition"
EXAMPLES = "examples"
COMPARISON = "comparison"
WHEN_TO_USE = "when_to_use"
GENERAL = "general"


def detect_answer_intent(question: str) -> str:
    low = question.lower()

    patterns = [
        (
            SUMMARY,
            r"\b(summarize|summarise|summary|overview|highlights|key points|brief)\b",
        ),
        (
            EXAMPLES,
            (
                r"\b("
                r"example|examples|for instance|for example|such as|"
                r"sample use cases?"
                r")\b"
            ),
        ),
        (
            COMPARISON,
            r"\b(compare|comparison|difference|different|vs|versus)\b",
        ),
        (
            WHEN_TO_USE,
            (
                r"\b("
                r"when should|when to use|when do|use cases?|best for|"
                r"recommended|suitable"
                r")\b"
            ),
        ),
        (
            DEFINITION,
            (
                r"\b("
                r"what is|what do you mean by|define|definition|explain|"
                r"in simple terms|meaning of"
                r")\b"
            ),
        ),
    ]

    for intent, pattern in patterns:
        if re.search(pattern, low):
            return intent
    return GENERAL


def infer_chunk_section_type(text: str) -> str:
    low = text.lower()
    patterns = [
        (
            EXAMPLES,
            r"\b(for example|for instance|such as|e\.g\.|examples include)\b",
        ),
        (
            COMPARISON,
            r"\b(difference|different|whereas|while|compared to|versus|vs)\b",
        ),
        (
            WHEN_TO_USE,
            (
                r"\b("
                r"used for|best for|recommended|suitable|should be used|"
                r"ideal for"
                r")\b"
            ),
        ),
        (DEFINITION, r"\b(is a|is an|refers to|defined as|definition|means)\b"),
        (SUMMARY, r"\b(summary|overview|highlights|key points|in short)\b"),
    ]
    for section_type, pattern in patterns:
        if re.search(pattern, low):
            return section_type
    return GENERAL


def intent_signal_score(text: str, intent: str) -> float:
    low = text.lower()
    patterns = {
        SUMMARY: (
            r"\b(summary|overview|highlights|in short|key points)\b",
            0.08,
        ),
        DEFINITION: (
            r"\b(is a|is an|refers to|defined as|definition|means)\b",
            0.12,
        ),
        EXAMPLES: (
            r"\b(for example|for instance|such as|e\.g\.|examples include)\b",
            0.14,
        ),
        COMPARISON: (
            r"\b(difference|different|whereas|while|compared to|versus|vs)\b",
            0.12,
        ),
        WHEN_TO_USE: (
            r"\b(used for|best for|recommended|suitable|should be used|ideal for|when)\b",
            0.12,
        ),
    }
    marker = patterns.get(intent)
    if marker is None:
        return 0.0
    pattern, weight = marker
    return weight if re.search(pattern, low) else 0.0


def section_type_boost(section_type: str, intent: str) -> float:
    if not section_type or section_type == GENERAL or intent == GENERAL:
        return 0.0
    if section_type == intent:
        return 0.18
    return 0.0
