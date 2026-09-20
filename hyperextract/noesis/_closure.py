"""Closure invariants for Noesis event closure components (contract section 6)."""

from .models import NoesisAtom


class _SemanticFailure(Exception):
    """Internal marker raised when one deterministic semantic rule fails."""

    def __init__(self, rule: str, field_path: str | None = None):
        super().__init__(rule)
        self.rule = rule
        self.field_path = field_path


def _check_pos_sequence(atoms: list[NoesisAtom]) -> None:
    positions = [atom.pos for atom in atoms]
    if positions != list(range(1, len(atoms) + 1)):
        raise _SemanticFailure("pos_sequence", "atoms")


def _find_unique_root(atoms: list[NoesisAtom]) -> NoesisAtom:
    roots = [atom for atom in atoms if atom.role == "predicate" and atom.target_occ is None]
    if len(roots) != 1:
        raise _SemanticFailure("unique_root", "atoms")
    return roots[0]


def _check_closure(atoms: list[NoesisAtom], root: NoesisAtom) -> None:
    """Every non-root atom must reach the unique root along target_occ edges.

    Walking each atom to the root detects self references, missing targets,
    cycles and islands in one pass; cross-component targets are impossible
    because each component is validated in isolation.
    """
    by_pos = {atom.pos: atom for atom in atoms}
    for index, atom in enumerate(atoms):
        if atom.pos == root.pos:
            continue
        field_path = f"atoms[{index}].target_occ"
        visited: set[int] = set()
        current = atom
        while current.pos != root.pos:
            if current.pos in visited:
                raise _SemanticFailure("closure_cycle", field_path)
            visited.add(current.pos)
            target = current.target_occ
            if target is None:
                raise _SemanticFailure("closure_missing_target", field_path)
            if target == current.pos:
                raise _SemanticFailure("closure_self_reference", field_path)
            if target not in by_pos:
                raise _SemanticFailure("closure_unknown_target", field_path)
            current = by_pos[target]


def _check_edge_roles(atoms: list[NoesisAtom]) -> None:
    """Section 6.2 edge rules: each role may only point at specific roles.

    agent/patient point at their SPO predicate, modifiers point at the
    agent/predicate/patient they modify, and a subordinate predicate points at
    either the superior SPO predicate or the argument modified by its clause.
    """
    by_pos = {atom.pos: atom for atom in atoms}
    for index, atom in enumerate(atoms):
        if atom.target_occ is None:
            continue
        target_role = by_pos[atom.target_occ].role
        field_path = f"atoms[{index}].target_occ"
        if atom.role in ("agent", "patient"):
            if target_role != "predicate":
                raise _SemanticFailure("edge_target_role_invalid", field_path)
        elif atom.role in ("modifier", "predicate") and target_role not in (
            "agent",
            "predicate",
            "patient",
        ):
            raise _SemanticFailure("edge_target_role_invalid", field_path)
