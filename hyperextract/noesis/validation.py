"""Deterministic semantic validation for Noesis Stage 1 event closure components."""

import re
import unicodedata
from collections import Counter
from typing import Any

from ._closure import (
    _SemanticFailure,
    _check_closure,
    _check_edge_roles,
    _check_pos_sequence,
    _find_unique_root,
)
from ._rule_track import _check_rule_template
from ._tree_track import _check_tree_structure
from .models import (
    ExtractionAlert,
    FactComponent,
    HypothesisComponent,
    NoesisAtom,
    NoesisExtraction,
    ValidationResult,
)

_RULE_MESSAGES = {
    "empty_text": "atom text is empty after normalization",
    "pos_sequence": "atoms pos must be consecutive integers starting from 1",
    "unique_root": "component must have exactly one root predicate with target_occ=null",
    "closure_unknown_target": "target_occ points to a pos missing in this component",
    "closure_self_reference": "target_occ points to the atom itself",
    "closure_cycle": "target_occ chain forms a cycle",
    "closure_missing_target": "non-root atom must have a non-null target_occ",
    "edge_target_role_invalid": "target_occ violates the role pointing rules of section 6.2",
    "atom_type_role_mismatch": "atom type is invalid for its role (section 5.3)",
    "source_text_violation": "atom text is not present in the source text (section 5.2)",
    "anaphora_antecedent_missing": "resolved=true atom has no antecedent with the same text (section 5.5)",
    "tree_root_predicate": "tree.predicate must equal the root predicate atom text",
    "tree_arg_mismatch": "tree arguments do not match atoms by role, target and count",
    "tree_modifier_mismatch": "tree modifiers do not match modifier atoms by placement and count",
    "tree_predicate_no_atom": "nested/conditional predicate has no matching subordinate predicate atom",
    "tree_predicate_unaccounted": "a subordinate predicate atom is missing from nested/conditional",
    "implied_text_not_entity": "implied argument text is not an entity atom text (section 7.1)",
    "tree_new_word": "tree introduces text missing from atoms",
    "rule_premise_mismatch": "rule_template premise does not match an atom (text, type, role)",
    "rule_conclusion_mismatch": "rule_template conclusion does not project the event closure",
    "rule_template_new_word": "rule_template condition introduces text missing from atoms",
}

_WHITESPACE_RUN = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Frozen deterministic normalization (section 5.2).

    Full-width characters fold to half-width, punctuation is removed entirely
    and whitespace collapses to single spaces.
    """
    folded: list[str] = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            folded.append(chr(code - 0xFEE0))
        else:
            folded.append(char)
    stripped = "".join(
        char for char in folded if not unicodedata.category(char).startswith("P")
    )
    return _WHITESPACE_RUN.sub(" ", stripped).strip()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    return value


def _normalize_component(
    component: FactComponent | HypothesisComponent,
) -> FactComponent | HypothesisComponent:
    data = _normalize_value(component.model_dump())
    for index, atom_data in enumerate(data["atoms"]):
        if not atom_data["text"]:
            raise _SemanticFailure("empty_text", f"atoms[{index}].text")
    return type(component).model_validate(data)


def _check_atom_type_role(atoms: list[NoesisAtom]) -> None:
    """Predicates are P, agents/patients are E, modifiers either (section 5.3)."""
    for index, atom in enumerate(atoms):
        valid = (
            (atom.role == "predicate" and atom.type == "P")
            or (atom.role in ("agent", "patient") and atom.type == "E")
            or atom.role == "modifier"
        )
        if not valid:
            raise _SemanticFailure("atom_type_role_mismatch", f"atoms[{index}].type")


def _check_source_containment(atoms: list[NoesisAtom], normalized_source: str) -> None:
    """Every atom text must appear in the normalized source (section 5.2)."""
    for index, atom in enumerate(atoms):
        if atom.text not in normalized_source:
            raise _SemanticFailure("source_text_violation", f"atoms[{index}].text")


def _check_anaphora_antecedents(atoms: list[NoesisAtom], text_counts: Counter) -> None:
    """resolved=true requires another atom with the same text (section 5.5)."""
    for index, atom in enumerate(atoms):
        if atom.resolved is True and text_counts[atom.text] < 2:
            raise _SemanticFailure("anaphora_antecedent_missing", f"atoms[{index}].resolved")


def _check_component(
    component: FactComponent | HypothesisComponent,
    normalized_source: str,
    text_counts: Counter,
) -> None:
    atoms = component.atoms
    _check_pos_sequence(atoms)
    root = _find_unique_root(atoms)
    _check_closure(atoms, root)
    _check_edge_roles(atoms)
    _check_atom_type_role(atoms)
    _check_source_containment(atoms, normalized_source)
    _check_anaphora_antecedents(atoms, text_counts)
    _check_tree_structure(component.tree, atoms, root)
    if isinstance(component, HypothesisComponent):
        _check_rule_template(component, root)


def _drop_alert(component_index: int, failure: _SemanticFailure) -> ExtractionAlert:
    details: dict[str, Any] = {"component_index": component_index, "rule": failure.rule}
    if failure.field_path is not None:
        details["field_path"] = failure.field_path
    return ExtractionAlert(
        stage="hyper_extract",
        alert_code="invalid_component_dropped",
        severity="warning",
        message=_RULE_MESSAGES[failure.rule],
        details=details,
    )


def validate_components(raw: object, *, source_text: str) -> ValidationResult:
    """Validate raw extraction output against the authoritative contract.

    Schema failures raise the pydantic ``ValidationError`` so the extractor can
    retry; semantic failures drop the offending component with a warning alert
    while keeping the remaining components. The source text is only used for
    containment checks and never leaks into alerts.
    """
    extraction = NoesisExtraction.model_validate(raw)
    normalized_source = _normalize_text(source_text)

    staged: list[FactComponent | HypothesisComponent | None] = []
    normalization_failures: list[tuple[int, _SemanticFailure]] = []
    for index, component in enumerate(extraction.root):
        try:
            staged.append(_normalize_component(component))
        except _SemanticFailure as failure:
            staged.append(None)
            normalization_failures.append((index, failure))

    # Antecedent pool: texts of every schema-valid atom, collected before drops
    # so a resolved=true atom may reference an entity of another component.
    text_counts: Counter[str] = Counter()
    for component in staged:
        if component is not None:
            text_counts.update(atom.text for atom in component.atoms)

    kept: list[FactComponent | HypothesisComponent] = []
    alerts: list[ExtractionAlert] = []
    for index, failure in normalization_failures:
        alerts.append(_drop_alert(index, failure))
    for index, component in enumerate(staged):
        if component is None:
            continue
        try:
            _check_component(component, normalized_source, text_counts)
        except _SemanticFailure as failure:
            alerts.append(_drop_alert(index, failure))
            continue
        kept.append(component)
    return ValidationResult(components=kept, alerts=alerts)
