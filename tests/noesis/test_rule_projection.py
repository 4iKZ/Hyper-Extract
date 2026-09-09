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
    """04A §4: the rule track is not bound by tree ⊆ atoms — premise entries
    may be source-backed composite expressions absent from the atom set."""

    def test_premise_composite_expression_passes(self):
        rule = valid_sun_rule()
        rule["premise"].append({"text": "打酱油", "type": "P", "role": "modifier"})

        result = validate([sun_hypothesis(rule)], "太阳每天从东边升起，打酱油")

        assert len(result.components) == 1
        assert result.alerts == []

    def test_premise_role_describes_rule_not_event_edge(self):
        rule = valid_sun_rule()
        rule["premise"] = [{"text": "太阳", "type": "P", "role": "predicate"}]

        result = validate([sun_hypothesis(rule)], "太阳每天从东边升起")

        assert len(result.components) == 1
        assert result.alerts == []

    def test_premise_absent_from_source_is_dropped(self):
        rule = valid_sun_rule()
        rule["premise"].append({"text": "月亮", "type": "E", "role": "agent"})

        result = validate([sun_hypothesis(rule)], "太阳每天从东边升起")

        assert_dropped(result)


class TestConclusionProjection:
    """Conclusion literals must project matching roles in the closure."""

    def test_conclusion_predicate_not_root_dropped(self):
        rule = valid_sun_rule()
        rule["conclusion"]["predicate"] = "太阳"

        result = validate([sun_hypothesis(rule)], "太阳每天从东边升起")

        assert_dropped(result)

    def test_conclusion_role_swap_is_dropped(self):
        rule = valid_apple_rule()
        rule["conclusion"]["agent"] = ["苹果"]
        rule["conclusion"]["patient"] = ["小明"]
        rule["conclusion"]["modifier"] = ["小明"]

        result = validate([apple_hypothesis(rule)], "小明每天吃苹果")

        assert_dropped(result)


class TestConditionProjection:
    """04A §4: conditions may keep negated composite expressions like
    「NOT 打酱油」 when the expression is grounded in the source."""

    def test_source_backed_composite_condition_passes(self):
        rule = valid_sun_rule()
        rule["condition"] = ["NOT 东边升起"]

        result = validate([sun_hypothesis(rule)], "太阳每天从东边升起")

        assert len(result.components) == 1
        assert result.alerts == []

    def test_condition_from_any_atom_passes(self):
        rule = valid_sun_rule()
        rule["condition"] = ["太阳"]

        result = validate([sun_hypothesis(rule)], "太阳每天从东边升起")

        assert len(result.components) == 1
        assert result.alerts == []

    def test_condition_absent_from_source_is_dropped(self):
        rule = valid_sun_rule()
        rule["condition"] = ["火星爆炸"]

        result = validate([sun_hypothesis(rule)], "太阳每天从东边升起")

        assert_dropped(result)


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
