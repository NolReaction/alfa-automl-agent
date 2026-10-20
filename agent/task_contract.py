"""Conservative metric declarations for the offline tabular task contract.

This is not a natural-language task interpreter. Supported descriptions contain
an explicit English or Russian declaration such as ``Primary metric: Macro F1``
or ``Метрика оценки — RMSE``. A metric heading followed by its name also works.
The name must lead the declaration; supplementary metrics belong on a separate
line or sentence. Unknown, conflicting, and missing declarations are errors.
"""

from __future__ import annotations

import re


_DECLARATION = re.compile(
    r"^(?P<label>(?:(?:primary|main|evaluation|scoring|target)\s+)?metric"
    r"|(?:(?:основная|главная|целевая)\s+)?метрика(?:\s+оценки)?)"
    r"(?=$|\s|[:=—–-])(?P<value>.*)$", re.IGNORECASE,
)
_ALIASES = (
    ("Macro F1", r"(?:macro[\s_-]*(?:averaged[\s_-]*)?f[\s_-]*1(?:[\s_-]*score)?"
     r"|f[\s_-]*1[\s_-]+macro)"),
    ("RMSE", r"(?:rmse|root\s+mean\s+squared\s+error)"),
)
_METRIC_TOKEN = re.compile(
    r"\b(?:rmse|rmsle|mse|mae|mape|smape|msle|r2|auc|roc[\s_-]*auc|"
    r"accuracy|precision|recall|log[\s_-]*loss|(?:macro|micro|weighted)[\s_-]*f[\s_-]*1|"
    r"f[\s_-]*1|root\s+mean\s+squared\s+error)\b", re.IGNORECASE,
)
_FORMAT = (
    "Use an explicit declaration in task.md: 'Primary metric: Macro F1' "
    "for classification or 'Metric: RMSE' for regression "
    "(also supported: 'Основная метрика' / 'Метрика оценки')."
)


def _description_lines(text: str) -> list[str]:
    """Remove Markdown decorations and code examples, keeping line boundaries."""
    lines = []
    fence = None
    # HTML comments can contain examples or stale declarations.
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    for raw in text.splitlines():
        stripped = raw.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            current = marker.group(1)
            if fence is None:
                fence = current
            elif current[0] == fence[0] and len(current) >= len(fence):
                fence = None
            continue
        if fence is not None or raw.startswith("    ") or raw.startswith("\t"):
            continue
        line = re.sub(r"^\s*(?:#{1,6}\s+|[-+*]\s+)", "", raw)
        line = line.replace("**", "").replace("__", "").replace("`", "").strip()
        lines.append(line)
    return lines


def _metric_value(value: str) -> str:
    value = re.sub(r"^(?:\s*[:=—–-]\s*)+", "", value).strip()
    value = re.sub(r"^(?:is|является|выбрана)\s+", "", value, flags=re.IGNORECASE)
    # A later sentence may explain secondary metrics; it is not the declaration.
    declaration = value.split(".", 1)[0].strip()
    for name, alias in _ALIASES:
        matched = re.match(rf"{alias}(?![\w])", declaration, flags=re.IGNORECASE)
        if matched is None:
            continue
        remainder = declaration[matched.end():]
        # Permit the standard expanded name with its acronym in parentheses.
        if name == "RMSE":
            remainder = re.sub(r"^\s*\(rmse\)", "", remainder, flags=re.IGNORECASE)
        if (_METRIC_TOKEN.search(remainder)
                or re.search(r"[/|]|\b(?:or|либо|или)\b", remainder, flags=re.IGNORECASE)):
            raise ValueError("Ambiguous primary metric declaration in task.md. " + _FORMAT)
        return name
    raise ValueError("Unsupported or unrecognized primary metric in task.md. " + _FORMAT)


def validate_task_metric(task_text: str, classification: bool) -> dict:
    """Resolve an explicitly declared supported metric, otherwise fail clearly.

    Supplementary lines such as ``Additional metrics: Accuracy, Weighted F1``
    are ignored. Multiple primary/evaluation declarations must agree, including
    declarations appearing in different Markdown sections.
    """
    if not isinstance(task_text, str) or type(classification) is not bool:
        raise ValueError("Task description must be text and classification must be a boolean.")
    lines = _description_lines(task_text)
    declarations = []
    for index, line in enumerate(lines):
        match = _DECLARATION.match(line)
        if match is None:
            continue
        value = match["value"].strip()
        if not value.strip(":=—–- "):
            value = next((item for item in lines[index + 1:] if item), "")
        declarations.append(_metric_value(value))
    if not declarations:
        raise ValueError("Missing explicit primary metric declaration in task.md. " + _FORMAT)
    if len(set(declarations)) != 1:
        raise ValueError("Conflicting primary metric declarations in task.md. " + _FORMAT)
    expected = "Macro F1" if classification else "RMSE"
    if declarations[0] != expected:
        raise ValueError(
            f"Task declares {declarations[0]}, but this offline "
            f"{'classification' if classification else 'regression'} mode supports {expected}."
        )
    return {"name": expected, "higher_is_better": classification, "source": "task.md"}
