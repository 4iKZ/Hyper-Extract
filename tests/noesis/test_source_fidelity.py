"""Source fidelity checks: containment, anaphora antecedent and punctuation."""

from hyperextract.noesis import validate_components


def atom(pos, text, type_, role, target_occ, resolved=None):
    return {
        "pos": pos,
        "text": text,
        "type": type_,
        "role": role,
        "target_occ": target_occ,
        "resolved": resolved,
    }


def arg(text, modifier=None, implied=False):
    return {"text": text, "modifier": list(modifier or []), "implied": implied}


def tree(predicate, agent=(), patient=(), modifier=(), nested=(), conditional=()):
    return {
        "predicate": predicate,
        "agent": list(agent),
        "patient": list(patient),
        "modifier": list(modifier),
        "nested": list(nested),
        "conditional": list(conditional),
    }


def fact(atoms, tree_):
    return {"utterance_type": "fact", "atoms": atoms, "tree": tree_}


def mother_fact():
    return fact(
        atoms=[
            atom(1, "妈妈", "E", "agent", 2),
            atom(2, "买", "P", "predicate", None),
            atom(3, "苹果", "E", "patient", 2),
        ],
        tree_=tree("买", agent=[arg("妈妈")], patient=[arg("苹果")]),
    )


def validate(raw, source_text):
    return validate_components(raw, source_text=source_text)


def assert_dropped(result):
    assert result.components == []
    assert len(result.alerts) == 1
    assert result.alerts[0].alert_code == "invalid_component_dropped"
    assert result.alerts[0].severity == "warning"
    assert result.alerts[0].stage == "hyper_extract"


class TestSourceContainment:
    """Every atom text must come from the source text (requirement 5.2)."""

    def test_fabricated_atoms_dropped(self):
        component = fact(
            atoms=[
                atom(1, "爸爸", "E", "agent", 2),
                atom(2, "卖", "P", "predicate", None),
                atom(3, "香蕉", "E", "patient", 2),
            ],
            tree_=tree("卖", agent=[arg("爸爸")], patient=[arg("香蕉")]),
        )

        result = validate([component], "妈妈买苹果")

        assert_dropped(result)

    def test_non_substring_atom_text_dropped(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "P", "predicate", None),
                atom(3, "苹果派", "E", "patient", 2),
            ],
            tree_=tree("买", agent=[arg("妈妈")], patient=[arg("苹果派")]),
        )

        result = validate([component], "妈妈买苹果")

        assert_dropped(result)

    def test_atoms_from_source_pass(self):
        result = validate([mother_fact()], "妈妈买苹果")

        assert len(result.components) == 1
        assert result.alerts == []


class TestAnaphoraAntecedent:
    """resolved=true requires an antecedent atom with the same text (requirement 5.5)."""

    def test_resolved_true_without_antecedent_dropped(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 2, resolved=True),
            ],
            tree_=tree("买", agent=[arg("妈妈")], patient=[arg("苹果")]),
        )

        result = validate([component], "妈妈买苹果。")

        assert_dropped(result)

    def test_resolved_true_antecedent_in_other_component_passes(self):
        component_b = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2, resolved=True),
                atom(2, "吃", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 2, resolved=True),
            ],
            tree_=tree("吃", agent=[arg("妈妈")], patient=[arg("苹果")]),
        )

        result = validate([mother_fact(), component_b], "妈妈买苹果。她吃了它。")

        assert len(result.components) == 2
        assert result.alerts == []

    def test_resolved_true_antecedent_in_same_component_passes(self):
        component = fact(
            atoms=[
                atom(1, "小明", "E", "agent", 2),
                atom(2, "揍", "P", "predicate", None),
                atom(3, "小明", "E", "patient", 2, resolved=True),
            ],
            tree_=tree("揍", agent=[arg("小明")], patient=[arg("小明")]),
        )

        result = validate([component], "小明揍了自己。")

        assert len(result.components) == 1
        assert result.alerts == []


class TestPunctuationNormalization:
    """Frozen rule: punctuation is removed, not mapped to ASCII (requirement 5.2)."""

    def test_punctuation_removed_from_stored_text(self):
        component = fact(
            atoms=[
                atom(1, "妈妈。", "E", "agent", 2),
                atom(2, "买", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 2),
            ],
            tree_=tree("买", agent=[arg("妈妈。")], patient=[arg("苹果")]),
        )

        result = validate([component], "妈妈。买苹果")

        assert len(result.components) == 1
        assert result.components[0].atoms[0].text == "妈妈"
        assert result.components[0].tree.agent[0].text == "妈妈"

    def test_text_spanning_removed_punctuation_passes(self):
        component = fact(
            atoms=[atom(1, "买苹果", "P", "predicate", None)],
            tree_=tree("买苹果"),
        )

        result = validate([component], "买，苹果")

        assert len(result.components) == 1
        assert result.alerts == []

    def test_punctuation_only_atom_text_dropped(self):
        component = fact(
            atoms=[atom(1, "。", "P", "predicate", None)],
            tree_=tree("。"),
        )

        result = validate([component], "买苹果。")

        assert_dropped(result)

    def test_full_width_text_matched_against_half_width_source(self):
        component = fact(
            atoms=[
                atom(1, "ＡＢＣ", "E", "agent", 2),
                atom(2, "买", "P", "predicate", None),
            ],
            tree_=tree("买", agent=[arg("ＡＢＣ")]),
        )

        result = validate([component], "ABC买苹果")

        assert len(result.components) == 1
        assert result.components[0].atoms[0].text == "ABC"
