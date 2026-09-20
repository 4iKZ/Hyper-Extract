"""Rule-template projection checks for the latest Stage 1 contract.

The rule track may compose source-backed expressions such as ``打酱油`` and
``NOT 打酱油`` even when those exact strings are not individual atoms.  That
exception is deliberately narrow: it does not permit the model to introduce
entities, actions, conditions, or conclusions absent from the source.
"""

import re

from ._closure import _SemanticFailure
from .models import HypothesisComponent, NoesisAtom

_LOGICAL_PREFIX = re.compile(r"^(?:NOT)\s+", re.IGNORECASE)


def _source_backed(text: str, normalized_source: str) -> bool:
    """Return whether a rule expression is grounded in the input text.

    ``NOT`` is the one sanctioned synthetic operator in the authoritative
    examples.  The expression following it must still occur in the source.
    """
    literal = _LOGICAL_PREFIX.sub("", text).strip()
    return bool(literal) and literal in normalized_source


def _check_rule_template(
    component: HypothesisComponent, root: NoesisAtom, normalized_source: str
) -> None:
    """Require a source-grounded projection while permitting composition."""
    if component.rule_template.conclusion.predicate != root.text:
        raise _SemanticFailure("rule_conclusion_mismatch", "rule_template.conclusion.predicate")

    for index, premise in enumerate(component.rule_template.premise):
        if not _source_backed(premise.text, normalized_source):
            raise _SemanticFailure("rule_source_violation", f"rule_template.premise[{index}].text")

    conclusion = component.rule_template.conclusion
    for role in ("agent", "patient", "modifier"):
        allowed = {
            atom.text
            for atom in component.atoms
            if atom.role == role
        }
        for index, value in enumerate(getattr(conclusion, role)):
            if value not in allowed:
                raise _SemanticFailure(
                    "rule_projection_mismatch",
                    f"rule_template.conclusion.{role}[{index}]",
                )

    for index, condition in enumerate(component.rule_template.condition):
        if not _source_backed(condition, normalized_source):
            raise _SemanticFailure("rule_source_violation", f"rule_template.condition[{index}]")
