"""Tree/atoms dual-track structural accounting (contract sections 6.2 and 7)."""

from collections import Counter

from ._closure import _SemanticFailure
from .models import NoesisAtom, SemanticTree


def _check_tree_structure(
    tree: SemanticTree, atoms: list[NoesisAtom], root: NoesisAtom
) -> None:
    """Bind every tree level to its predicate atom and account for every atom.

    The tree is the nested human-readable track of the same closure as atoms
    (section 7.4): arguments, modifiers, markers and subordinate predicates
    must correspond occurrence-by-occurrence, in both directions.
    """
    modifiers_by_target: dict[int, list[str]] = {}
    for atom in atoms:
        if atom.role == "modifier" and atom.target_occ is not None:
            modifiers_by_target.setdefault(atom.target_occ, []).append(atom.text)
    entity_texts = {atom.text for atom in atoms if atom.type in ("E", "G")}
    atom_texts = {atom.text for atom in atoms}
    subordinates = [
        atom for atom in atoms if atom.role == "predicate" and atom.pos != root.pos
    ]
    consumed: set[int] = set()

    def bind(node: SemanticTree, parent: NoesisAtom, field: str) -> None:
        # A subordinate event either hangs from the superior predicate or acts
        # as a sentence-level modifier of one of that SPO's arguments.  Atoms
        # remain authoritative for which of those attachment points is used.
        allowed_targets = {parent.pos}
        allowed_targets.update(
            atom.pos
            for atom in atoms
            if atom.role in ("agent", "patient") and atom.target_occ == parent.pos
        )
        candidates = [
            atom
            for atom in subordinates
            if atom.text == node.predicate
            and atom.target_occ in allowed_targets
            and atom.pos not in consumed
        ]
        if not candidates:
            raise _SemanticFailure("tree_predicate_no_atom", field)
        bound = candidates[0]
        consumed.add(bound.pos)
        walk(node, bound, field)

    def check_arguments(role: str, arguments, level: NoesisAtom, field: str) -> None:
        role_atoms = [a for a in atoms if a.role == role and a.target_occ == level.pos]
        atom_counter = Counter(a.text for a in role_atoms)
        arg_counter = Counter(a.text for a in arguments if not a.implied)
        if arg_counter != atom_counter:
            raise _SemanticFailure("tree_arg_mismatch", f"{field}.{role}")
        for position, argument in enumerate(arguments):
            arg_field = f"{field}.{role}[{position}]"
            if argument.implied:
                if argument.text not in entity_texts:
                    raise _SemanticFailure("implied_text_not_entity", f"{arg_field}.text")
                for index, modifier in enumerate(argument.modifier):
                    if modifier not in atom_texts:
                        raise _SemanticFailure(
                            "tree_new_word", f"{arg_field}.modifier[{index}]"
                        )
        for text in arg_counter:
            positions = [a.pos for a in role_atoms if a.text == text]
            arg_modifiers = Counter()
            for argument in arguments:
                if not argument.implied and argument.text == text:
                    arg_modifiers.update(argument.modifier)
            atom_modifiers = Counter()
            for pos in positions:
                atom_modifiers.update(modifiers_by_target.get(pos, []))
            if arg_modifiers != atom_modifiers:
                raise _SemanticFailure("tree_modifier_mismatch", f"{field}.{role}")

    def walk(node: SemanticTree, level: NoesisAtom, field: str) -> None:
        for role, arguments in (("agent", node.agent), ("patient", node.patient)):
            check_arguments(role, arguments, level, field)
        markers = [b.marker for b in node.conditional if b.marker is not None]
        expected = Counter(modifiers_by_target.get(level.pos, []))
        if Counter(node.modifier) + Counter(markers) != expected:
            raise _SemanticFailure("tree_modifier_mismatch", f"{field}.modifier")
        for index, nested in enumerate(node.nested):
            bind(nested, level, f"{field}.nested[{index}]")
        for index, branch_node in enumerate(node.conditional):
            bind(branch_node.event, level, f"{field}.conditional[{index}].event")

    if tree.predicate != root.text:
        raise _SemanticFailure("tree_root_predicate", "tree.predicate")
    walk(tree, root, "tree")
    if len(consumed) != len(subordinates):
        raise _SemanticFailure("tree_predicate_unaccounted", "atoms")
