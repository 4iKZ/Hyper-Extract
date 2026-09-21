"""Source fidelity checks: containment, anaphora antecedent and punctuation."""

import pytest

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
    """Sentence punctuation is trimmed without corrupting literal identity."""

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

    def test_text_spanning_source_punctuation_is_not_treated_as_literal_match(self):
        component = fact(
            atoms=[atom(1, "买苹果", "P", "predicate", None)],
            tree_=tree("买苹果"),
        )

        result = validate([component], "买，苹果")

        assert_dropped(result)
        assert result.alerts[0].details["rule"] == "source_text_violation"

    @pytest.mark.parametrize(
        "literal",
        [
            "order-api-7d9c",
            "app_settle_service.py",
            "/api/v1/health",
            "AIOPS-20260916-049",
            "1.6 GiB",
            "0.2%",
            "user_id",
            "created_at",
            "user-center > mysql-users",
        ],
    )
    def test_semantic_internal_punctuation_is_preserved(self, literal):
        component = fact(
            atoms=[atom(1, literal, "P", "predicate", None)],
            tree_=tree(literal),
        )

        result = validate([component], f"检查结果：{literal}。")

        assert len(result.components) == 1
        assert result.alerts == []
        assert result.components[0].atoms[0].text == literal
        assert result.components[0].tree.predicate == literal

    def test_punctuation_changing_hallucination_is_rejected(self):
        component = fact(
            atoms=[atom(1, "orderapi7d9c", "P", "predicate", None)],
            tree_=tree("orderapi7d9c"),
        )

        result = validate([component], "order-api-7d9c")

        assert_dropped(result)
        assert result.alerts[0].details["rule"] == "source_text_violation"

    def test_containment_allows_whitespace_normalization(self):
        component = fact(
            atoms=[
                atom(1, "40分钟", "E", "modifier", 2),
                atom(2, "观察", "P", "predicate", None),
            ],
            tree_=tree("观察", modifier=["40分钟"]),
        )

        result = validate([component], "观察 40 分钟")

        assert len(result.components) == 1
        assert result.alerts == []
        assert result.components[0].atoms[0].text == "40分钟"

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


class TestEntityGranularityGate:
    """Proposition-shaped E atoms reject their component and request one retry."""

    def test_proposition_shaped_entity_is_dropped_and_requests_retry(self):
        clause = "连续观察 40 分钟后内存稳定在 1.6 GiB"
        component = fact(
            atoms=[
                atom(1, "复核结果", "E", "agent", 2),
                atom(2, "显示", "P", "predicate", None),
                atom(3, clause, "E", "patient", 2),
            ],
            tree_=tree("显示", agent=[arg("复核结果")], patient=[arg(clause)]),
        )

        result = validate([component], f"复核结果显示，{clause}。")

        assert result.components == []
        assert result.retry_needed is True
        assert [alert.alert_code for alert in result.alerts] == ["invalid_component_dropped"]
        assert result.alerts[0].details == {
            "component_index": 0,
            "rule": "entity_clause_shape",
            "atom_positions": [3],
            "signals": ["entity_contains_clause_cue"],
        }

    @pytest.mark.parametrize(
        "literal",
        [
            "2026年09月10日 09:22",
            "210 万条时间序列",
            "order-api-production-canary-7d9c",
            "kubectl describe pod order-api-7c8d9f6b5-x2vqn",
            "harbor.internal/payment-gateway:v1.88.3",
            "java.util.concurrent.TimeoutException",
            "/api/order/list?page=1",
        ],
    )
    def test_time_quantity_and_long_identifier_do_not_alert_by_length_alone(self, literal):
        component = fact(
            atoms=[
                atom(1, literal, "E", "modifier", 2),
                atom(2, "记录", "P", "predicate", None),
            ],
            tree_=tree("记录", modifier=[literal]),
        )

        result = validate([component], f"{literal}记录")

        assert len(result.components) == 1
        assert result.alerts == []
