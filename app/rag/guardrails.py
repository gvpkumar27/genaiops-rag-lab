"""Prompt-security guardrails for user question risk detection."""

import json
import re
from functools import lru_cache
from pathlib import Path

from app.config import settings

_BASE_EDUCATIONAL_INTENT = [
    r"\b(what is|define|explain|difference|example|how does|how to detect)\b",
]

_BASE_HARMFUL_INTENT = [
    r"\b(how to|steps to|guide to|instructions to)\b",
    r"\b(build|make|create|assemble)\b.{0,30}\b(bomb|malware|ransomware|exploit)\b",
    r"\b(hack|breach|exploit|ddos|phish|steal|hot[- ]?wire)\b",
    r"\b(bypass|disable)\b.{0,30}\b(auth|mfa|security|guardrails|immobilizer)\b",
]

_BASE_PATTERNS: dict[str, list[str]] = {
    "prompt_injection": [
        r"\bignore (all )?(previous|prior|above) (instructions|prompts)\b",
        r"\boverride (the )?(system|developer) (prompt|instructions)\b",
        r"\breveal (the )?(system|developer) (prompt|message)\b",
        r"\bprint (the )?(hidden|internal) (prompt|instructions)\b",
    ],
    "jailbreak": [
        r"\bjailbreak\b",
        r"\bdo anything now\b",
        r"\b(bypass|disable) (safety|guardrails|restrictions)\b",
        r"\bpretend to be unrestricted\b",
    ],
    "information_exfiltration": [
        r"\b(show|reveal|dump|leak|extract)\b.{0,40}\b(api[- ]?key|token|secret)\b",
        r"\bprint\b.{0,40}\b(environment variables|env vars|dotenv|\.env)\b",
        r"\bshow\b.{0,40}\bprivate\b.{0,20}\bdata\b",
    ],
    "phishing_or_social_engineering": [
        r"\bwrite\b.{0,30}\b(phishing|credential-harvesting)\b",
        r"\bconvince\b.{0,40}\bshare\b.{0,20}\b(password|otp|token)\b",
        r"\bimpersonate\b.{0,30}\b(it admin|support|bank|hr)\b",
    ],
    "tool_or_system_abuse": [
        r"\brm -rf\b",
        r"\b(powershell|bash|cmd)\b.{0,40}\b(delete|disable|exfiltrate)\b",
        r"\bcurl\b.{0,40}\b(attacker|webhook|pastebin)\b",
    ],
}


def _resolve_guardrails_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path


def _compile_regex_list(values: list[str], context: str) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for idx, pattern in enumerate(values):
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            raise RuntimeError(f"Invalid regex in {context}[{idx}]: {exc}") from exc
    return compiled


def _load_private_rules() -> dict:
    if not settings.GUARDRAILS_FILE:
        return {}

    path = _resolve_guardrails_path(settings.GUARDRAILS_FILE)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Failed to read GUARDRAILS_FILE at '{path}': {exc}") from exc

    if not text:
        raise RuntimeError(f"GUARDRAILS_FILE is empty: '{path}'")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in GUARDRAILS_FILE '{path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("GUARDRAILS_FILE must be a JSON object")

    return payload


def _parse_private_sections(payload: dict) -> tuple[list[str], dict, list[str]]:
    disable_cats = payload.get("disable_baseline_categories", [])
    private_patterns = payload.get("patterns", {})
    private_edu = payload.get("educational_intent", [])

    if not isinstance(disable_cats, list):
        raise RuntimeError("disable_baseline_categories must be a list")
    if not isinstance(private_patterns, dict):
        raise RuntimeError("patterns must be a JSON object")
    if not isinstance(private_edu, list):
        raise RuntimeError("educational_intent must be a list")

    return disable_cats, private_patterns, private_edu


def _parse_blocked_terms(payload: dict) -> list[str]:
    blocked_terms = payload.get("blocked_terms", [])
    if not isinstance(blocked_terms, list):
        raise RuntimeError("blocked_terms must be a list")
    return [term for term in blocked_terms if isinstance(term, str)]


def _parse_sensitive_terms(payload: dict) -> list[str]:
    sensitive_terms = payload.get("sensitive_terms", [])
    if not isinstance(sensitive_terms, list):
        raise RuntimeError("sensitive_terms must be a list")
    return [term for term in sensitive_terms if isinstance(term, str)]


def _merge_private_rules(
    raw_patterns: dict[str, list[str]],
    educational: list[str],
    payload: dict,
) -> None:
    disable_cats, private_patterns, private_edu = _parse_private_sections(payload)

    for cat in disable_cats:
        if isinstance(cat, str):
            raw_patterns.pop(cat, None)

    for category, pattern_list in private_patterns.items():
        if not isinstance(category, str) or not isinstance(pattern_list, list):
            raise RuntimeError("Each patterns entry must map string -> list")
        raw_patterns.setdefault(category, []).extend(
            [pattern for pattern in pattern_list if isinstance(pattern, str)]
        )

    educational.extend([pattern for pattern in private_edu if isinstance(pattern, str)])


def _normalize_term(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _tokenize_terms(text: str) -> set[str]:
    return {
        _normalize_term(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
    }


@lru_cache(maxsize=1)
def _compiled_rule_set() -> tuple[
    dict[str, list[re.Pattern[str]]],
    list[re.Pattern[str]],
    list[re.Pattern[str]],
    set[str],
    set[str],
]:
    raw_patterns: dict[str, list[str]] = {k: list(v) for k, v in _BASE_PATTERNS.items()}
    educational = list(_BASE_EDUCATIONAL_INTENT)
    harmful_intent = list(_BASE_HARMFUL_INTENT)
    blocked_terms: set[str] = set()
    sensitive_terms: set[str] = set()

    if settings.GUARDRAILS_FILE:
        payload = _load_private_rules()
        _merge_private_rules(raw_patterns, educational, payload)
        blocked_terms = {_normalize_term(term) for term in _parse_blocked_terms(payload)}
        blocked_terms.discard("")
        sensitive_terms = {_normalize_term(term) for term in _parse_sensitive_terms(payload)}
        sensitive_terms.discard("")
        private_harmful_intent = payload.get("harmful_intent", [])
        if not isinstance(private_harmful_intent, list):
            raise RuntimeError("harmful_intent must be a list")
        harmful_intent.extend(
            [pattern for pattern in private_harmful_intent if isinstance(pattern, str)]
        )

    compiled_patterns = {
        category: _compile_regex_list(values, f"patterns.{category}")
        for category, values in raw_patterns.items()
    }
    compiled_edu = _compile_regex_list(educational, "educational_intent")
    compiled_harmful_intent = _compile_regex_list(harmful_intent, "harmful_intent")
    return (
        compiled_patterns,
        compiled_edu,
        compiled_harmful_intent,
        blocked_terms,
        sensitive_terms,
    )


def validate_guardrails_config() -> None:
    """Fail fast on invalid private guardrails configuration."""
    _compiled_rule_set()


def _blocked_term_match(question_terms: set[str], blocked_terms: set[str]) -> bool:
    return bool(blocked_terms and (question_terms & blocked_terms))


def _classify_sensitive_harmful(
    text: str,
    question_terms: set[str],
    educational_patterns: list[re.Pattern[str]],
    harmful_intent_patterns: list[re.Pattern[str]],
    sensitive_terms: set[str],
) -> dict | None:
    looks_educational = any(pattern.search(text) for pattern in educational_patterns)
    has_harmful_intent = any(
        pattern.search(text) for pattern in harmful_intent_patterns
    )
    has_sensitive_topic = bool(sensitive_terms and (question_terms & sensitive_terms))

    if has_harmful_intent and not looks_educational:
        return {
            "blocked": True,
            "categories": ["harmful_content"],
            "reason": "harmful_intent_detected",
        }
    if has_sensitive_topic and has_harmful_intent:
        return {
            "blocked": True,
            "categories": ["harmful_content"],
            "reason": "sensitive_topic_with_harmful_intent",
        }
    if has_sensitive_topic and not has_harmful_intent:
        return {
            "blocked": False,
            "categories": ["sensitive_topic"],
            "reason": "sensitive_topic_non_actionable",
        }
    return None


def _match_pattern_categories(
    text: str,
    patterns: dict[str, list[re.Pattern[str]]],
) -> list[str]:
    matches: list[str] = []
    for category, category_patterns in patterns.items():
        for pattern in category_patterns:
            if pattern.search(text):
                matches.append(category)
                break
    return matches


def analyze_question_risk(question: str) -> dict:
    """Return risk analysis for a user question.

    Educational or analytical prompts about attacks are not blocked by default.
    """
    (
        patterns,
        educational_patterns,
        harmful_intent_patterns,
        blocked_terms,
        sensitive_terms,
    ) = _compiled_rule_set()
    text = question.strip()
    question_terms = _tokenize_terms(text)
    if _blocked_term_match(question_terms, blocked_terms):
        return {
            "blocked": True,
            "categories": ["blocked_term"],
            "reason": "blocked_term_match",
        }
    sensitive_result = _classify_sensitive_harmful(
        text=text,
        question_terms=question_terms,
        educational_patterns=educational_patterns,
        harmful_intent_patterns=harmful_intent_patterns,
        sensitive_terms=sensitive_terms,
    )
    if sensitive_result:
        return sensitive_result

    matches = _match_pattern_categories(text, patterns)

    if not matches:
        return {
            "blocked": False,
            "categories": [],
            "reason": "",
        }

    unique_categories = sorted(set(matches))
    looks_educational = any(pattern.search(text) for pattern in educational_patterns)
    blocked = not looks_educational

    return {
        "blocked": blocked,
        "categories": unique_categories,
        "reason": "potential_prompt_attack" if blocked else "educational_security_query",
    }
