"""Deterministic semantic validation for Noesis Stage 1 event closure components.

04A §5 splits the pipeline into a bounded repair layer that runs on the raw
model output *before* the strict Pydantic/semantic validation:

* role values outside the four-value enum demote to ``modifier`` + warning;
* an externally extracted component containing a ``G`` (Geneme) atom is
  dropped with a non-blocking alert — G is system-constructed only;
* pos gaps/duplicates are never renumbered: the component is dropped and the
  whole input is flagged for one retry;
* competing root predicates are arbitrated when the tree (or the direct
  argument count) uniquely selects the core predicate — non-clause verbs
  demote to ``modifier``, true subordinate-clause predicates never demote;
* a missing root predicate is repaired only when the unique ``type=P`` atom
  and the tree root agree;
* a wrong ``target_occ`` is repaired only when the tree and the existing pos
  values uniquely determine the target; self references, cycles, targets
  outside the component and ambiguous cases retry then drop.

The closure invariants themselves are never relaxed: after the bounded
repairs every component still passes the full strict validation below.
"""

import re
import unicodedata
from collections import Counter
from typing import Any

from ._closure import (
    _check_closure,
    _check_edge_roles,
    _check_pos_sequence,
    _find_unique_root,
    _SemanticFailure,
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
    "rule_conclusion_mismatch": "rule_template conclusion does not project the event closure",
    "rule_projection_mismatch": "rule_template role projection does not match the event closure",
    "rule_source_violation": "rule_template introduces content missing from the source text",
}

_REPAIR_MESSAGES = {
    "external_geneme_rejected": (
        "external component contains a G (Geneme) atom; G is system-constructed only"
    ),
    "role_demoted_to_modifier": (
        "atom role outside the four-value enum demoted to modifier before strict validation"
    ),
    "coordinate_predicate_demoted": (
        "competing root predicates arbitrated from the tree; the non-clause verb demoted to modifier"
    ),
    "missing_predicate_repaired": (
        "root predicate role repaired from the unique type=P atom and the tree root"
    ),
    "target_occ_repaired": (
        "target_occ repaired to the unique target determined by the semantic tree"
    ),
}

_WHITESPACE_RUN = re.compile(r"\s+")

_VALID_ROLES = frozenset({"agent", "patient", "predicate", "modifier"})


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


# ---------------------------------------------------------------------------
# Bounded repair layer (04A §5) — raw pre-validation before the strict models
# ---------------------------------------------------------------------------


def _has_external_geneme(component: dict) -> bool:
    """G is a legal stored type but never a legal external extraction."""
    atoms = component.get("atoms")
    if isinstance(atoms, list):
        for entry in atoms:
            if isinstance(entry, dict) and entry.get("type") == "G":
                return True
    rule = component.get("rule_template")
    if isinstance(rule, dict):
        premises = rule.get("premise")
        if isinstance(premises, list):
            for premise in premises:
                if isinstance(premise, dict) and premise.get("type") == "G":
                    return True
    return False


def _remove_tree_argument(tree_node: Any, text: Any) -> bool:
    """Drop the first non-implied argument entry matching ``text`` from any
    tree level's agent/patient lists and re-attach it as a modifier string of
    that level, keeping the dual track consistent with the demoted atom."""
    if not isinstance(text, str):
        return False
    for level in _iter_tree_levels(tree_node):
        for role in ("agent", "patient"):
            entries = level.get(role)
            if not isinstance(entries, list):
                continue
            for position, arg_entry in enumerate(entries):
                if (
                    isinstance(arg_entry, dict)
                    and arg_entry.get("text") == text
                    and not arg_entry.get("implied")
                ):
                    del entries[position]
                    modifiers = level.get("modifier")
                    if isinstance(modifiers, list):
                        modifiers.append(text)
                    return True
    return False


def _demote_invalid_roles(component: dict, index: int, alerts: list) -> None:
    """Roles outside the four-value enum demote to modifier; never guessed.
    The tree argument entry for the same text follows the demotion so the
    dual track stays consistent."""
    atoms = component.get("atoms")
    tree_node = component.get("tree")
    if not isinstance(atoms, list):
        return
    demoted = False
    for entry in atoms:
        if isinstance(entry, dict):
            role = entry.get("role")
            if isinstance(role, str) and role not in _VALID_ROLES:
                entry["role"] = "modifier"
                demoted = True
                if isinstance(tree_node, dict):
                    _remove_tree_argument(tree_node, entry.get("text"))
    if demoted:
        alerts.append(_repair_alert(index, "role_demoted_to_modifier"))


def _pos_sequence_broken(component: dict) -> bool:
    """pos must be 1..N consecutive integers; schema-level pos problems defer
    to the strict layer (they trigger the schema retry instead)."""
    atoms = component.get("atoms")
    if not isinstance(atoms, list):
        return False
    positions: list[int] = []
    for entry in atoms:
        if not isinstance(entry, dict):
            continue
        pos = entry.get("pos")
        if isinstance(pos, bool) or not isinstance(pos, int):
            return False
        positions.append(pos)
    return positions != list(range(1, len(positions) + 1))


def _collect_subordinate_tree_predicates(tree_node: Any, acc: set) -> None:
    if not isinstance(tree_node, dict):
        return
    for nested in tree_node.get("nested") or []:
        if isinstance(nested, dict):
            predicate = nested.get("predicate")
            if isinstance(predicate, str):
                acc.add(predicate)
            _collect_subordinate_tree_predicates(nested, acc)
    for branch in tree_node.get("conditional") or []:
        if isinstance(branch, dict):
            _collect_subordinate_tree_predicates(branch.get("event"), acc)


def _arbitrate_root_predicate(component: dict, index: int, alerts: list) -> tuple[dict | None, bool]:
    """Competing root predicates: keep the tree-selected core predicate (ties
    broken by direct argument count), demote non-clause verbs to modifier,
    never demote true subordinate-clause predicates (they get their parent
    target repaired instead). Returns (component, retry_needed)."""
    atoms = component.get("atoms")
    tree_node = component.get("tree")
    if not isinstance(atoms, list) or not isinstance(tree_node, dict):
        return component, False

    roots = [
        entry
        for entry in atoms
        if isinstance(entry, dict)
        and entry.get("role") == "predicate"
        and entry.get("target_occ") is None
    ]
    if len(roots) <= 1:
        return component, False

    tree_predicate = tree_node.get("predicate")
    if not isinstance(tree_predicate, str):
        return None, True
    candidates = [entry for entry in roots if entry.get("text") == tree_predicate]
    if len(candidates) > 1:
        argument_counts = {entry["pos"]: 0 for entry in candidates}
        for entry in atoms:
            if isinstance(entry, dict) and entry.get("target_occ") in argument_counts:
                argument_counts[entry["target_occ"]] += 1
        best = max(argument_counts.values())
        candidates = [entry for entry in candidates if argument_counts[entry["pos"]] == best]
    if len(candidates) != 1:
        return None, True

    keeper = candidates[0]
    subordinate_texts: set = set()
    _collect_subordinate_tree_predicates(tree_node, subordinate_texts)
    demoted_texts: list = []
    for root in roots:
        if root is keeper:
            continue
        if root.get("text") in subordinate_texts:
            # A true subordinate-clause predicate: repair its parent target,
            # never demote it to modifier.
            root["target_occ"] = keeper["pos"]
        else:
            root["role"] = "modifier"
            root["target_occ"] = keeper["pos"]
            demoted_texts.append(root.get("text"))
    if demoted_texts:
        modifiers = tree_node.get("modifier")
        if isinstance(modifiers, list):
            modifiers.extend(text for text in demoted_texts if isinstance(text, str))
        alerts.append(_repair_alert(index, "coordinate_predicate_demoted"))
    return component, False


def _repair_missing_predicate(component: dict, index: int, alerts: list) -> tuple[dict | None, bool]:
    """No root predicate: repair the role only when the unique type=P atom and
    the tree root agree; never invent an atom."""
    atoms = component.get("atoms")
    tree_node = component.get("tree")
    if not isinstance(atoms, list) or not isinstance(tree_node, dict):
        return component, False

    has_root = any(
        isinstance(entry, dict)
        and entry.get("role") == "predicate"
        and entry.get("target_occ") is None
        for entry in atoms
    )
    if has_root:
        return component, False

    tree_predicate = tree_node.get("predicate")
    if not isinstance(tree_predicate, str):
        return None, True
    matches = [
        entry
        for entry in atoms
        if isinstance(entry, dict)
        and entry.get("type") == "P"
        and entry.get("text") == tree_predicate
    ]
    if len(matches) != 1:
        return None, True
    matches[0]["role"] = "predicate"
    matches[0]["target_occ"] = None
    alerts.append(_repair_alert(index, "missing_predicate_repaired"))
    return component, False


def _iter_tree_levels(tree_node: Any):
    if not isinstance(tree_node, dict):
        return
    yield tree_node
    for nested in tree_node.get("nested") or []:
        yield from _iter_tree_levels(nested)
    for branch in tree_node.get("conditional") or []:
        if isinstance(branch, dict):
            yield from _iter_tree_levels(branch.get("event"))


def _tree_determined_target(
    tree_node: Any, by_pos: dict[int, dict], text: Any, role: str
) -> int | None:
    """The unique predicate pos the tree says this (text, role) argument
    belongs to, or None when the tree cannot determine it uniquely."""
    if not isinstance(text, str):
        return None
    level_predicates: list[str] = []
    for level in _iter_tree_levels(tree_node):
        predicate = level.get("predicate")
        if not isinstance(predicate, str):
            continue
        for entry in level.get(role) or []:
            if (
                isinstance(entry, dict)
                and entry.get("text") == text
                and not entry.get("implied")
            ):
                level_predicates.append(predicate)
                break
    if len(set(level_predicates)) != 1:
        return None
    predicate_text = level_predicates[0]
    positions = [
        pos
        for pos, entry in by_pos.items()
        if entry.get("role") == "predicate" and entry.get("text") == predicate_text
    ]
    if len(positions) != 1:
        return None
    return positions[0]


def _tree_determined_modifier_target(
    tree_node: Any, by_pos: dict[int, dict], text: Any
) -> int | None:
    """Return the unique target encoded by the tree for one modifier text."""
    if not isinstance(text, str):
        return None
    targets: list[int] = []
    for level in _iter_tree_levels(tree_node):
        predicate = level.get("predicate")
        predicate_positions = [
            pos
            for pos, entry in by_pos.items()
            if entry.get("role") == "predicate" and entry.get("text") == predicate
        ]
        if len(predicate_positions) != 1:
            continue
        predicate_pos = predicate_positions[0]
        targets.extend(predicate_pos for value in level.get("modifier") or [] if value == text)
        targets.extend(
            predicate_pos
            for branch in level.get("conditional") or []
            if isinstance(branch, dict) and branch.get("marker") == text
        )
        for role in ("agent", "patient"):
            for argument in level.get(role) or []:
                if not isinstance(argument, dict) or argument.get("implied"):
                    continue
                if text not in (argument.get("modifier") or []):
                    continue
                argument_positions = [
                    pos
                    for pos, entry in by_pos.items()
                    if entry.get("role") == role
                    and entry.get("text") == argument.get("text")
                    and entry.get("target_occ") == predicate_pos
                ]
                if len(argument_positions) == 1:
                    targets.append(argument_positions[0])
    return targets[0] if len(targets) == 1 else None


def _repair_targets(component: dict, index: int, alerts: list) -> tuple[dict | None, bool]:
    """Repair uniquely tree-determined argument/modifier targets.

    Self references, cycles, targets outside the component, and ambiguous
    tree projections retry then drop.
    """
    atoms = component.get("atoms")
    tree_node = component.get("tree")
    if not isinstance(atoms, list) or not isinstance(tree_node, dict):
        return component, False

    by_pos: dict[int, dict] = {}
    for entry in atoms:
        if isinstance(entry, dict):
            pos = entry.get("pos")
            if isinstance(pos, int) and not isinstance(pos, bool):
                by_pos[pos] = entry
    if len(by_pos) != len(atoms):
        return component, False  # schema-level atom shapes: strict layer

    for pos, entry in by_pos.items():
        target = entry.get("target_occ")
        if target is None:
            continue
        if target == pos:
            return None, True  # self reference
        if target not in by_pos:
            return None, True  # cross-component / unknown target

    for start in by_pos:
        visited: set[int] = set()
        current = start
        while True:
            target = by_pos[current].get("target_occ")
            if target is None:
                break
            if target in visited:
                return None, True  # cycle
            visited.add(target)
            current = target

    repaired = False
    for entry in by_pos.values():
        role = entry.get("role")
        if role in ("agent", "patient"):
            new_target = _tree_determined_target(
                tree_node, by_pos, entry.get("text"), role
            )
        elif role == "modifier":
            new_target = _tree_determined_modifier_target(
                tree_node, by_pos, entry.get("text")
            )
        else:
            continue
        target = entry.get("target_occ")
        if new_target is None:
            # Keep an already role-valid edge for strict tree accounting;
            # otherwise this malformed edge is not safely repairable.
            target_role = by_pos[target].get("role") if target in by_pos else None
            valid_existing = (
                role in ("agent", "patient") and target_role == "predicate"
            ) or (
                role == "modifier" and target_role in ("agent", "predicate", "patient")
            )
            if valid_existing:
                continue
            return None, True
        if target != new_target:
            entry["target_occ"] = new_target
            repaired = True
    if repaired:
        alerts.append(_repair_alert(index, "target_occ_repaired"))
    return component, False


def _repair_alert(component_index: int, repair_code: str) -> ExtractionAlert:
    return ExtractionAlert(
        stage="hyper_extract",
        alert_code=repair_code,
        severity="warning",
        message=_REPAIR_MESSAGES[repair_code],
        details={"component_index": component_index},
    )


def _pre_drop_alert(component_index: int, rule: str) -> ExtractionAlert:
    return ExtractionAlert(
        stage="hyper_extract",
        alert_code="invalid_component_dropped",
        severity="warning",
        message=_RULE_MESSAGES[rule],
        details={"component_index": component_index, "rule": rule},
    )


def _pre_validate(raw: Any) -> tuple[Any, list[ExtractionAlert], bool]:
    """Run the bounded repairs on the raw output; returns the corrected raw,
    the repair/drop alerts, and whether the caller should retry once."""
    if not isinstance(raw, list):
        return raw, [], False
    alerts: list[ExtractionAlert] = []
    retry_needed = False
    kept: list[Any] = []
    for index, component in enumerate(raw):
        if not isinstance(component, dict):
            kept.append(component)  # strict schema layer rejects it
            continue
        if _has_external_geneme(component):
            alerts.append(_repair_alert(index, "external_geneme_rejected"))
            continue
        _demote_invalid_roles(component, index, alerts)
        if _pos_sequence_broken(component):
            retry_needed = True
            alerts.append(_pre_drop_alert(index, "pos_sequence"))
            continue
        component, retry = _arbitrate_root_predicate(component, index, alerts)
        if component is None:
            retry_needed = True
            alerts.append(_pre_drop_alert(index, "unique_root"))
            continue
        retry_needed = retry_needed or retry
        component, retry = _repair_missing_predicate(component, index, alerts)
        if component is None:
            retry_needed = True
            alerts.append(_pre_drop_alert(index, "unique_root"))
            continue
        retry_needed = retry_needed or retry
        component, retry = _repair_targets(component, index, alerts)
        if component is None:
            retry_needed = True
            alerts.append(_pre_drop_alert(index, "closure_unknown_target"))
            continue
        retry_needed = retry_needed or retry
        kept.append(component)
    return kept, alerts, retry_needed


# ---------------------------------------------------------------------------
# Strict validation layer (unchanged invariants)
# ---------------------------------------------------------------------------


def _check_atom_type_role(atoms: list[NoesisAtom]) -> None:
    """Predicates are P; E/G fill entity roles; modifiers accept any stored type."""
    for index, atom in enumerate(atoms):
        valid = (
            (atom.role == "predicate" and atom.type == "P")
            or (atom.role in ("agent", "patient") and atom.type in ("E", "G"))
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
        _check_rule_template(component, root, normalized_source)


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
    retry; bounded repairs (04A §5) run on the raw payload first and record
    their own warnings; semantic failures drop the offending component with a
    warning alert while keeping the remaining components. ``retry_needed``
    tells the extractor to retry the whole input once before accepting the
    drops. The source text is only used for containment checks and never
    leaks into alerts.
    """
    corrected_raw, repair_alerts, retry_needed = _pre_validate(raw)
    extraction = NoesisExtraction.model_validate(corrected_raw)
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
    alerts: list[ExtractionAlert] = list(repair_alerts)
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
    return ValidationResult(components=kept, alerts=alerts, retry_needed=retry_needed)
