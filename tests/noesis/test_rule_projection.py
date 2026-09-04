"""Rule template structural projection checks."""

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


def hypothesis(atoms, tree_, rule_template):
    return {
        "utterance_type": "hypothesis",
        "atoms": atoms,
        "tree": tree_,
        "rule_template": rule_template,
    }


def validate(raw, source_text):
    return validate_components(raw, source_text=source_text)


def assert_dropped(result):
    assert result.components == []
    assert len(result.alerts) == 1
    assert result.alerts[0].alert_code == "invalid_component_dropped"
    assert result.alerts[0].severity == "warning"
    assert result.alerts[0].stage == "hyper_extract"


def sun_hypothesis(rule_template):
    return hypothesis(
        atoms=[
            atom(1, "太阳", "E", "agent", 3),
            atom(2, "每天", "E", "modifier", 3),
            atom(3, "升起", "P", "predicate", None),
            atom(4, "东边", "E", "modifier", 3),
        ],
        tree_=tree("升起", agent=[arg("太阳")], modifier=["每天", "东边"]),
        rule_template=rule_template,
    )


def valid_sun_rule():
    return {
        "premise": [{"text": "太阳", "type": "E", "role": "agent"}],
        "conclusion": {
            "predicate": "升起",
            "agent": ["太阳"],
            "patient": [],
            "modifier": ["东边"],
        },
        "condition": ["每天"],
    }


def apple_hypothesis(rule_template):
    return hypothesis(
        atoms=[
            atom(1, "小明", "E", "agent", 3),
            atom(2, "每天", "E", "modifier", 3),
            atom(3, "吃", "P", "predicate", None),
            atom(4, "苹果", "E", "patient", 3),
        ],
        tree_=tree("吃", agent=[arg("小明")], patient=[arg("苹果")], modifier=["每天"]),
        rule_template=rule_template,
    )


def valid_apple_rule():
    return {
        "premise": [{"text": "小明", "type": "E", "role": "agent"}],
        "conclusion": {
            "predicate": "吃",
            "agent": ["小明"],
            "patient": ["苹果"],
            "modifier": [],
        },
        "condition": ["每天"],
    }


class TestPremiseProjection:
    """premise entries must match an atom's (text, type, role) triple (8.2)."""

    def test_premise_type_mismatch_dropped(self):
        rule = valid_sun_rule()
        rule["premise"] = [{"text": "太阳", "type": "P", "role": "agent"}]

        result = validate([sun_hypothesis(rule)], "太阳每天从东边升起")

        assert_dropped(result)

    def test_premise_role_mismatch_dropped(self):
        rule = valid_sun_rule()
        rule["premise"] = [{"text": "太阳", "type": "E", "role": "predicate"}]

        result = validate([sun_hypothesis(rule)], "太阳每天从东边升起")

        assert_dropped(result)

    def test_premise_type_mismatch_with_patient_role_dropped(self):
        rule = valid_apple_rule()
        rule["premise"] = [{"text": "苹果", "type": "P", "role": "patient"}]

        result = validate([apple_hypothesis(rule)], "小明每天吃苹果")

        assert_dropped(result)


class TestConclusionProjection:
    """conclusion must project the root event with role-matched atoms (8.3)."""

    def test_conclusion_predicate_not_root_dropped(self):
        rule = valid_sun_rule()
        rule["conclusion"]["predicate"] = "太阳"

        result = validate([sun_hypothesis(rule)], "太阳每天从东边升起")

        assert_dropped(result)

    def test_conclusion_agent_role_mismatch_dropped(self):
        rule = valid_apple_rule()
        rule["conclusion"]["agent"] = ["苹果"]

        result = validate([apple_hypothesis(rule)], "小明每天吃苹果")

        assert_dropped(result)

    def test_conclusion_patient_role_mismatch_dropped(self):
        rule = valid_apple_rule()
        rule["conclusion"]["patient"] = ["小明"]

        result = validate([apple_hypothesis(rule)], "小明每天吃苹果")

        assert_dropped(result)

    def test_conclusion_modifier_role_mismatch_dropped(self):
        rule = valid_apple_rule()
        rule["conclusion"]["modifier"] = ["小明"]

        result = validate([apple_hypothesis(rule)], "小明每天吃苹果")

        assert_dropped(result)


class TestConditionProjection:
    """condition strings must come from atoms (8.4, role unrestricted)."""

    def test_condition_outside_atoms_dropped(self):
        rule = valid_sun_rule()
        rule["condition"] = ["有时"]

        result = validate([sun_hypothesis(rule)], "太阳每天从东边升起")

        assert_dropped(result)

    def test_condition_from_any_atom_passes(self):
        rule = valid_sun_rule()
        rule["condition"] = ["太阳"]

        result = validate([sun_hypothesis(rule)], "太阳每天从东边升起")

        assert len(result.components) == 1
        assert result.alerts == []


class TestValidProjection:
    """Full valid projections pass end to end."""

    def test_sun_projection_passes(self):
        result = validate([sun_hypothesis(valid_sun_rule())], "太阳每天从东边升起")

        assert len(result.components) == 1
        assert result.alerts == []

    def test_apple_projection_passes(self):
        result = validate([apple_hypothesis(valid_apple_rule())], "小明每天吃苹果")

        assert len(result.components) == 1
        assert result.alerts == []
