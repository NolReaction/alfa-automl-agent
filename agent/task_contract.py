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
    ("Weighted F1", r"(?:weighted[\s_-]*(?:averaged[\s_-]*)?f[\s_-]*1(?:[\s_-]*score)?"
     r"|f[\s_-]*1[\s_-]+weighted)"),
    ("Accuracy", r"accuracy(?:[\s_-]*score)?"),
    ("ROC AUC", r"roc[\s_-]*auc(?:[\s_-]*score)?"),
    ("LogLoss", r"(?:log[\s_-]*loss|logarithmic\s+loss|cross[\s_-]*entropy)"),
    ("MAE", r"(?:mae|mean\s+absolute\s+error)"),
    ("MSE", r"(?:mse|mean\s+squared\s+error)"),
    ("R2", r"(?:r[\s_-]*(?:2|²|\^2)(?:[\s_-]*score)?|coefficient\s+of\s+determination)"),
)
_METRIC_TOKEN = re.compile(
    r"\b(?:" + "|".join(alias for _, alias in _ALIASES)
    + r"|rmsle|mape|smape|msle|auc|precision|recall|micro[\s_-]*f[\s_-]*1|f[\s_-]*1)\b",
    re.IGNORECASE,
)
_FORMAT = (
    "Use an explicit declaration in task.md: 'Primary metric: Macro F1' "
    "for classification or 'Metric: RMSE' for regression "
    "(also supported: 'Основная метрика' / 'Метрика оценки')."
)


def _description_lines(text: str, *, preserve_values=False) -> list[str]:
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
        if not preserve_values:
            line = line.replace("**", "").replace("__", "").replace("`", "")
        lines.append(line.strip())
    return lines


def _metric_value(value: str, *, expanded=False) -> str:
    value = re.sub(r"^(?:\s*[:=—–-]\s*)+", "", value).strip()
    value = re.sub(r"^(?:is|является|выбрана)\s+", "", value, flags=re.IGNORECASE)
    # A later sentence may explain secondary metrics; it is not the declaration.
    declaration = value.split(".", 1)[0].strip()
    for name, alias in _ALIASES:
        matched = re.match(rf"{alias}(?![\w])", declaration, flags=re.IGNORECASE)
        if matched is None:
            continue
        if not expanded and name not in {"RMSE", "Macro F1"}:
            raise ValueError("Unsupported primary metric for the legacy baseline. " + _FORMAT)
        remainder = declaration[matched.end():]
        # An acronym or full expansion of this same metric is not a second
        # metric. Other aliases anywhere in the declaration remain ambiguous.
        parenthetical = re.match(r"^\s*\(([^()]*)\)", remainder)
        if (parenthetical is not None
                and re.fullmatch(alias, parenthetical[1].strip(), flags=re.IGNORECASE)):
            remainder = remainder[parenthetical.end():]
        if (_METRIC_TOKEN.search(remainder)
                or re.search(r"[/|]|\b(?:or|либо|или)\b", remainder, flags=re.IGNORECASE)):
            raise ValueError("Ambiguous primary metric declaration in task.md. " + _FORMAT)
        return name
    raise ValueError("Unsupported or unrecognized primary metric in task.md. " + _FORMAT)


def _declared_metric(task_text: str, *, expanded=False) -> str:
    if not isinstance(task_text, str):
        raise ValueError("Task description must be text")
    lines = _description_lines(task_text)
    declarations = []
    for index, line in enumerate(lines):
        match = _DECLARATION.match(line)
        if match is None:
            continue
        value = match["value"].strip()
        if not value.strip(":=—–- "):
            value = next((item for item in lines[index + 1:] if item), "")
        declarations.append(_metric_value(value, expanded=expanded))
    if not declarations:
        raise ValueError("Missing explicit primary metric declaration in task.md. " + _FORMAT)
    if len(set(declarations)) != 1:
        raise ValueError("Conflicting primary metric declarations in task.md. " + _FORMAT)
    return declarations[0]


def parse_task_metric(task_text: str):
    """Resolve the expanded trusted registry from an explicit primary declaration.

    Unqualified F1/AUC remain ambiguous. This parser never executes metric code
    supplied in the task or chooses a metric by guessing the target dtype.
    """
    from agent.metrics import get_metric

    return get_metric(_declared_metric(task_text, expanded=True))


def validate_task_metric(task_text: str, classification: bool) -> dict:
    """Resolve an explicitly declared supported metric, otherwise fail clearly.

    Supplementary lines such as ``Additional metrics: Accuracy, Weighted F1``
    are ignored. Multiple primary/evaluation declarations must agree, including
    declarations appearing in different Markdown sections.
    """
    if not isinstance(task_text, str) or type(classification) is not bool:
        raise ValueError("Task description must be text and classification must be a boolean.")
    declared = _declared_metric(task_text)
    expected = "Macro F1" if classification else "RMSE"
    if declared != expected:
        raise ValueError(
            f"Task declares {declared}, but this offline "
            f"{'classification' if classification else 'regression'} mode supports {expected}."
        )
    return {"name": expected, "higher_is_better": classification, "source": "task.md"}
