"""Rule template structural projection checks (contract section 8)."""

from ._closure import _SemanticFailure
from .models import HypothesisComponent, NoesisAtom


def _check_rule_template(component: HypothesisComponent, root: NoesisAtom) -> None:
    """The rule template must be a structural projection of atoms and tree.

    premise entries must match an atom's (text, type, role) triple; the
    conclusion predicate must be the root predicate; conclusion role lists
    must reference atoms of the matching role; conditions must come from
    atoms (section 8.4, role unrestricted).
    """
    atoms = component.atoms
    triples = {(atom.text, atom.type, atom.role) for atom in atoms}
    atom_texts = {atom.text for atom in atoms}
    texts_by_role: dict[str, set[str]] = {"agent": set(), "patient": set(), "modifier": set()}
    for atom in atoms:
        if atom.role in texts_by_role:
            texts_by_role[atom.role].add(atom.text)

    rule = component.rule_template
    for index, premise in enumerate(rule.premise):
        if (premise.text, premise.type, premise.role) not in triples:
            raise _SemanticFailure("rule_premise_mismatch", f"rule_template.premise[{index}]")

    conclusion = rule.conclusion
    if conclusion.predicate != root.text:
        raise _SemanticFailure(
            "rule_conclusion_mismatch", "rule_template.conclusion.predicate"
        )
    for field in ("agent", "patient", "modifier"):
        for index, value in enumerate(getattr(conclusion, field)):
            if value not in texts_by_role[field]:
                raise _SemanticFailure(
                    "rule_conclusion_mismatch", f"rule_template.conclusion.{field}[{index}]"
                )

    for index, condition in enumerate(rule.condition):
        if condition not in atom_texts:
            raise _SemanticFailure(
                "rule_template_new_word", f"rule_template.condition[{index}]"
            )
