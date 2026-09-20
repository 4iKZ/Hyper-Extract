"""Unit tests for Noesis component validation: closure, dual-track, anaphora (15.2-15.4)."""

import copy

import pytest

from hyperextract.noesis import validate_components

SOURCE_TEXT = (
    "昨天妈妈在超市买了苹果，小明吃了苹果。"
    "妈妈让小明打酱油，爸爸让小红洗碗。"
    "小明没写作业后揍了自己，他走了。"
    "如果下雨小明带伞。太阳每天从东边升起。"
    "小刚指着月亮说有时西边会落下西瓜。"
)


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


def branch(event, marker=None):
    return {"marker": marker, "event": event}


def fact(atoms, tree_):
    return {"utterance_type": "fact", "atoms": atoms, "tree": tree_}


def baseline_fact():
    """Minimal valid fact: 昨天妈妈买苹果."""
    return fact(
        atoms=[
            atom(1, "昨天", "E", "modifier", 3),
            atom(2, "妈妈", "E", "agent", 3),
            atom(3, "买", "P", "predicate", None),
            atom(4, "苹果", "E", "patient", 3),
        ],
        tree_=tree("买", agent=[arg("妈妈")], patient=[arg("苹果")], modifier=["昨天"]),
    )


def causative_fact():
    """Valid fact with one root and a subordinate predicate: 妈妈让小明打酱油."""
    return fact(
        atoms=[
            atom(1, "妈妈", "E", "agent", 2),
            atom(2, "让", "P", "predicate", None),
            atom(3, "小明", "E", "patient", 2),
            atom(4, "打", "P", "predicate", 2),
            atom(5, "酱油", "E", "patient", 4),
        ],
        tree_=tree(
            "让",
            agent=[arg("妈妈")],
            patient=[arg("小明")],
            nested=[tree("打", agent=[arg("小明", implied=True)], patient=[arg("酱油")])],
        ),
    )


def nested_fact():
    """Valid fact with subordinate event and resolved anaphor: 小明没写作业后揍了自己."""
    return fact(
        atoms=[
            atom(1, "小明", "E", "agent", 4),
            atom(2, "没写", "P", "predicate", 4),
            atom(3, "作业", "E", "patient", 2),
            atom(4, "揍", "P", "predicate", None),
            atom(5, "小明", "E", "patient", 4, resolved=True),
        ],
        tree_=tree(
            "揍",
            agent=[arg("小明")],
            patient=[arg("小明")],
            nested=[tree("没写", agent=[arg("小明", implied=True)], patient=[arg("作业")])],
        ),
    )


def sun_hypothesis(rule_template=None):
    """Valid hypothesis: 太阳每天从东边升起."""
    if rule_template is None:
        rule_template = {
            "premise": [{"text": "太阳", "type": "E", "role": "agent"}],
            "conclusion": {
                "predicate": "升起",
                "agent": ["太阳"],
                "patient": [],
                "modifier": ["东边"],
            },
            "condition": ["每天"],
        }
    return {
        "utterance_type": "hypothesis",
        "atoms": [
            atom(1, "太阳", "E", "agent", 3),
            atom(2, "每天", "E", "modifier", 3),
            atom(3, "升起", "P", "predicate", None),
            atom(4, "东边", "E", "modifier", 3),
        ],
        "tree": tree("升起", agent=[arg("太阳")], modifier=["每天", "东边"]),
        "rule_template": rule_template,
    }


def conditional_fact(marker="如果"):
    """Valid fact with a conditional branch: 如果下雨小明带伞."""
    if marker is None:
        atoms = [
            atom(1, "下雨", "P", "predicate", 4),
            atom(2, "小明", "E", "agent", 4),
            atom(3, "伞", "E", "patient", 4),
            atom(4, "带", "P", "predicate", None),
        ]
    else:
        atoms = [
            atom(1, "如果", "E", "modifier", 5),
            atom(2, "下雨", "P", "predicate", 5),
            atom(3, "小明", "E", "agent", 5),
            atom(4, "伞", "E", "patient", 5),
            atom(5, "带", "P", "predicate", None),
        ]
    return fact(
        atoms=atoms,
        tree_=tree(
            "带",
            agent=[arg("小明")],
            patient=[arg("伞")],
            conditional=[branch(tree("下雨"), marker=marker)],
        ),
    )


def validate(raw):
    return validate_components(raw, source_text=SOURCE_TEXT)


def assert_drop_alert(result, expected_count=1):
    assert len(result.alerts) == expected_count
    for alert in result.alerts:
        assert alert.alert_code == "invalid_component_dropped"
        assert alert.severity == "warning"
        assert alert.stage == "hyper_extract"


class TestClosureValidation:
    """target_occ closure invariants (requirement 15.2)."""

    def test_single_valid_root_passes(self):
        result = validate([baseline_fact()])

        assert len(result.components) == 1
        assert result.alerts == []

    def test_multiple_predicates_single_root_passes(self):
        result = validate([causative_fact()])

        assert len(result.components) == 1
        assert result.alerts == []

    def test_subordinate_predicate_connected_passes(self):
        result = validate([nested_fact()])

        assert len(result.components) == 1
        assert result.alerts == []

    def test_two_components_pos_each_start_at_one_passes(self):
        fact_a = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 2),
            ],
            tree_=tree("买", agent=[arg("妈妈")], patient=[arg("苹果")]),
        )
        fact_b = fact(
            atoms=[
                atom(1, "爸爸", "E", "agent", 2),
                atom(2, "洗", "P", "predicate", None),
                atom(3, "碗", "E", "patient", 2),
            ],
            tree_=tree("洗", agent=[arg("爸爸")], patient=[arg("碗")]),
        )

        result = validate([fact_a, fact_b])

        assert len(result.components) == 2
        assert result.alerts == []

    def test_shared_subject_coordinate_predicates_split_into_three_components(self):
        source_text = "小明坐在沙发上，吃着苹果，玩着苹果手机。"
        components = [
            fact(
                atoms=[
                    atom(1, "小明", "E", "agent", 2),
                    atom(2, "坐", "P", "predicate", None),
                    atom(3, "在沙发上", "E", "modifier", 2),
                ],
                tree_=tree("坐", agent=[arg("小明")], modifier=["在沙发上"]),
            ),
            fact(
                atoms=[
                    atom(1, "小明", "E", "agent", 2),
                    atom(2, "吃", "P", "predicate", None),
                    atom(3, "苹果", "E", "patient", 2),
                ],
                tree_=tree("吃", agent=[arg("小明")], patient=[arg("苹果")]),
            ),
            fact(
                atoms=[
                    atom(1, "小明", "E", "agent", 2),
                    atom(2, "玩", "P", "predicate", None),
                    atom(3, "苹果手机", "E", "patient", 2),
                ],
                tree_=tree("玩", agent=[arg("小明")], patient=[arg("苹果手机")]),
            ),
        ]

        result = validate_components(components, source_text=source_text)

        assert len(result.components) == 3
        assert result.alerts == []
        assert [component.tree.predicate for component in result.components] == [
            "坐",
            "吃",
            "玩",
        ]

    def test_two_roots_arbitrated_tree_selects_core_predicate(self):
        """04A §5: competing root predicates — the tree uniquely selects 买,
        the non-clause verb 吃 demotes to modifier with a warning."""
        component = fact(
            atoms=[
                atom(1, "买", "P", "predicate", None),
                atom(2, "吃", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 1),
            ],
            tree_=tree("买", patient=[arg("苹果")]),
        )

        result = validate([component])

        assert len(result.components) == 1
        atoms = result.components[0].model_dump()["atoms"]
        assert [entry["role"] for entry in atoms] == ["predicate", "modifier", "patient"]
        assert atoms[1]["target_occ"] == 1
        assert [alert.alert_code for alert in result.alerts] == ["coordinate_predicate_demoted"]

    def test_two_roots_unselectable_retries_then_drops(self):
        """The tree root matches neither competing predicate: not repairable,
        the component is flagged for a whole-input retry."""
        component = fact(
            atoms=[
                atom(1, "买", "P", "predicate", None),
                atom(2, "吃", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 1),
            ],
            tree_=tree("洗", patient=[arg("苹果")]),
        )

        result = validate([component])

        assert result.components == []
        assert result.retry_needed is True
        assert_drop_alert(result)

    def test_no_root_repaired_from_unique_p_atom(self):
        """04A §5: missing root predicate — the unique type=P atom that matches
        the tree root is repaired into the root predicate role."""
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "P", "predicate", 1),
            ],
            tree_=tree("买", agent=[arg("妈妈")]),
        )

        result = validate([component])

        assert len(result.components) == 1
        atoms = result.components[0].model_dump()["atoms"]
        assert atoms[1]["role"] == "predicate"
        assert atoms[1]["target_occ"] is None
        assert [alert.alert_code for alert in result.alerts] == ["missing_predicate_repaired"]

    def test_no_root_ambiguous_p_atoms_retries_then_drops(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "P", "modifier", 1),
                atom(3, "买", "P", "patient", 1),
            ],
            tree_=tree("买", agent=[arg("妈妈")]),
        )

        result = validate([component])

        assert result.components == []
        assert result.retry_needed is True
        assert_drop_alert(result)

    def test_target_missing_dropped(self):
        component = baseline_fact()
        component["atoms"][3]["target_occ"] = 99

        result = validate([component])

        assert result.components == []
        assert_drop_alert(result)

    def test_self_reference_dropped(self):
        component = baseline_fact()
        component["atoms"][0]["target_occ"] = 1

        result = validate([component])

        assert result.components == []
        assert_drop_alert(result)

    def test_two_node_cycle_dropped(self):
        component = fact(
            atoms=[
                atom(1, "买", "P", "predicate", None),
                atom(2, "妈妈", "E", "agent", 3),
                atom(3, "苹果", "E", "patient", 2),
            ],
            tree_=tree("买", agent=[arg("妈妈")], patient=[arg("苹果")]),
        )

        result = validate([component])

        assert result.components == []
        assert_drop_alert(result)

    def test_multi_node_cycle_dropped(self):
        component = fact(
            atoms=[
                atom(1, "买", "P", "predicate", None),
                atom(2, "妈妈", "E", "agent", 3),
                atom(3, "苹果", "E", "patient", 4),
                atom(4, "昨天", "E", "modifier", 2),
            ],
            tree_=tree("买", agent=[arg("妈妈")], patient=[arg("苹果")], modifier=["昨天"]),
        )

        result = validate([component])

        assert result.components == []
        assert_drop_alert(result)

    def test_island_atom_dropped(self):
        component = fact(
            atoms=[
                atom(1, "买", "P", "predicate", None),
                atom(2, "妈妈", "E", "agent", 1),
                atom(3, "昨天", "E", "modifier", 4),
                atom(4, "在超市", "E", "modifier", 3),
            ],
            tree_=tree("买", agent=[arg("妈妈")], modifier=["昨天", "在超市"]),
        )

        result = validate([component])

        assert result.components == []
        assert_drop_alert(result)

    def test_pos_duplicate_dropped(self):
        component = baseline_fact()
        component["atoms"][3]["pos"] = 3

        result = validate([component])

        assert result.components == []
        assert_drop_alert(result)

    def test_pos_gap_dropped(self):
        component = baseline_fact()
        component["atoms"][3]["pos"] = 5

        result = validate([component])

        assert result.components == []
        assert_drop_alert(result)

    def test_pos_not_starting_at_one_dropped(self):
        component = baseline_fact()
        for index, atom_ in enumerate(component["atoms"], start=2):
            atom_["pos"] = index
            if atom_["target_occ"] is not None:
                atom_["target_occ"] += 1

        result = validate([component])

        assert result.components == []
        assert_drop_alert(result)

    def test_cross_component_target_dropped(self):
        offender = baseline_fact()
        # pos 5 only exists in the other component.
        offender["atoms"][3]["target_occ"] = 5
        kept = causative_fact()

        result = validate([offender, kept])

        assert len(result.components) == 1
        assert result.components[0].model_dump() == kept
        assert_drop_alert(result)


class TestDualTrackConsistency:
    """tree/atoms dual-track consistency (requirement 15.3)."""

    def test_tree_root_predicate_matches_atoms_root(self):
        result = validate([baseline_fact()])

        assert len(result.components) == 1
        assert result.components[0].tree.predicate == "买"
        assert result.alerts == []

    def test_nested_predicate_matches_subordinate_atom(self):
        result = validate([nested_fact()])

        assert len(result.components) == 1
        assert result.components[0].tree.nested[0].predicate == "没写"
        assert result.alerts == []

    def test_conditional_marker_and_event_match_atoms(self):
        result = validate([conditional_fact(marker="如果")])

        assert len(result.components) == 1
        branch_ = result.components[0].tree.conditional[0]
        assert branch_.marker == "如果"
        assert branch_.event.predicate == "下雨"
        assert result.alerts == []

    def test_conditional_null_marker_passes(self):
        result = validate([conditional_fact(marker=None)])

        assert len(result.components) == 1
        assert result.components[0].tree.conditional[0].marker is None
        assert result.alerts == []

    @pytest.mark.parametrize(
        ("mutate",),
        [
            (lambda component: component["tree"]["patient"][0].update({"text": "西瓜"}),),
            (lambda component: component["tree"]["modifier"].append("在超市"),),
        ],
        ids=["new_patient_text", "new_modifier_text"],
    )
    def test_tree_new_word_dropped(self, mutate):
        component = baseline_fact()
        mutate(component)

        result = validate([component])

        assert result.components == []
        assert_drop_alert(result)

    def test_implied_agent_reuses_atom_text_passes(self):
        result = validate([nested_fact()])

        assert len(result.components) == 1
        implied_agent = result.components[0].tree.nested[0].agent[0]
        assert implied_agent.implied is True
        assert implied_agent.text == "小明"
        assert result.alerts == []

    def test_implied_agent_new_text_dropped(self):
        component = nested_fact()
        component["tree"]["nested"][0]["agent"][0]["text"] = "小刚"

        result = validate([component])

        assert result.components == []
        assert_drop_alert(result)

    def test_rule_template_source_backed_composition_not_literal_bound(self):
        """04A §4: the rule track is free of the tree ⊆ atoms constraint —
        composite premise expressions and negated conditions pass."""
        rule = {
            "premise": [
                {"text": "太阳", "type": "E", "role": "agent"},
                {"text": "打酱油", "type": "P", "role": "modifier"},
            ],
            "conclusion": {
                "predicate": "升起",
                "agent": ["太阳"],
                "patient": [],
                "modifier": ["东边"],
            },
            "condition": ["每天", "NOT 打酱油"],
        }

        result = validate_components(
            [sun_hypothesis(rule_template=rule)],
            source_text="太阳每天从东边升起，打酱油",
        )

        assert len(result.components) == 1
        assert result.alerts == []

    def test_rule_template_conclusion_predicate_not_root_dropped(self):
        rule = {
            "premise": [{"text": "太阳", "type": "E", "role": "agent"}],
            "conclusion": {
                "predicate": "落下",
                "agent": ["太阳"],
                "patient": [],
                "modifier": ["东边"],
            },
            "condition": ["每天"],
        }

        result = validate([sun_hypothesis(rule_template=rule)])

        assert result.components == []
        assert_drop_alert(result)


class TestAnaphora:
    """resolved tri-state anaphora rules (requirement 15.4)."""

    def test_resolved_true_replaces_with_unique_antecedent(self):
        component = fact(
            atoms=[
                atom(1, "小明", "E", "agent", 2),
                atom(2, "揍", "P", "predicate", None),
                atom(3, "小明", "E", "patient", 2, resolved=True),
            ],
            tree_=tree("揍", agent=[arg("小明")], patient=[arg("小明")]),
        )

        result = validate([component])

        assert len(result.components) == 1
        assert result.alerts == []
        resolved_atom = result.components[0].atoms[2]
        assert resolved_atom.resolved is True
        assert resolved_atom.text == "小明"

    def test_resolved_false_keeps_pronoun(self):
        component = fact(
            atoms=[
                atom(1, "他", "E", "agent", 2, resolved=False),
                atom(2, "走", "P", "predicate", None),
            ],
            tree_=tree("走", agent=[arg("他")]),
        )

        result = validate([component])

        assert len(result.components) == 1
        assert result.alerts == []
        pronoun_atom = result.components[0].atoms[0]
        assert pronoun_atom.resolved is False
        assert pronoun_atom.text == "他"

    def test_ordinary_atoms_resolved_null(self):
        result = validate([baseline_fact()])

        assert len(result.components) == 1
        assert all(atom_.resolved is None for atom_ in result.components[0].atoms)

    def test_repeated_text_keeps_distinct_pos(self):
        result = validate([nested_fact()])

        assert len(result.components) == 1
        atoms_ = result.components[0].atoms
        occurrences = [atom_ for atom_ in atoms_ if atom_.text == "小明"]

        assert [atom_.pos for atom_ in occurrences] == [1, 5]
        assert [atom_.pos for atom_ in atoms_] == [1, 2, 3, 4, 5]


class TestDropAlerts:
    """Alert shape for dropped components (requirement 10.4/10.5)."""

    def test_drop_alert_fields(self):
        component = baseline_fact()
        component["atoms"][0]["target_occ"] = 1

        result = validate([component])

        assert result.components == []
        assert_drop_alert(result)
        alert = result.alerts[0]
        assert isinstance(alert.message, str) and alert.message
        assert isinstance(alert.details, dict)

    def test_alert_details_do_not_leak_full_source_text(self):
        component = baseline_fact()
        component["atoms"][0]["target_occ"] = 1

        result = validate([component])

        assert_drop_alert(result)
        assert SOURCE_TEXT not in str(result.alerts[0].details)

    def test_all_dropped_returns_empty_components_with_alerts(self):
        first = baseline_fact()
        first["atoms"][0]["target_occ"] = 1
        second = baseline_fact()
        second["atoms"][1]["target_occ"] = 2

        result = validate([first, second])

        assert result.components == []
        assert len(result.alerts) >= 1
        assert all(alert.alert_code == "invalid_component_dropped" for alert in result.alerts)

    def test_mixed_valid_and_invalid_keeps_valid_only(self):
        invalid = baseline_fact()
        invalid["atoms"][0]["target_occ"] = 1
        valid = causative_fact()

        result = validate([invalid, valid])

        assert len(result.components) == 1
        assert result.components[0].model_dump() == copy.deepcopy(valid)
        assert_drop_alert(result)


class TestBoundedRepairs:
    """04A §5: the raw pre-validation repair layer and its boundaries."""

    def test_external_geneme_dropped_with_non_blocking_alert(self):
        component = baseline_fact()
        component["atoms"][3]["type"] = "G"

        result = validate([component])

        assert result.components == []
        assert [alert.alert_code for alert in result.alerts] == ["external_geneme_rejected"]
        assert all(alert.severity == "warning" for alert in result.alerts)
        assert result.retry_needed is False

    def test_external_geneme_in_rule_premise_dropped(self):
        rule = {
            "premise": [{"text": "太阳", "type": "G", "role": "agent"}],
            "conclusion": {"predicate": "升起", "agent": [], "patient": [], "modifier": []},
            "condition": [],
        }

        result = validate([sun_hypothesis(rule_template=rule)])

        assert result.components == []
        assert [alert.alert_code for alert in result.alerts] == ["external_geneme_rejected"]

    def test_empty_array_is_normal_zero_alert(self):
        result = validate([])

        assert result.components == []
        assert result.alerts == []
        assert result.retry_needed is False

    def test_role_outside_enum_demoted_to_modifier_with_warning(self):
        component = baseline_fact()
        component["atoms"][1]["role"] = "subject"

        result = validate([component])

        assert len(result.components) == 1
        assert result.components[0].atoms[1].role == "modifier"
        assert [alert.alert_code for alert in result.alerts] == ["role_demoted_to_modifier"]
        assert result.retry_needed is False

    def test_pos_gap_flags_whole_input_retry(self):
        component = baseline_fact()
        component["atoms"][3]["pos"] = 5

        result = validate([component])

        assert result.components == []
        assert result.retry_needed is True
        assert result.alerts[0].alert_code == "invalid_component_dropped"
        assert result.alerts[0].details["rule"] == "pos_sequence"

    def test_pos_duplicate_flags_whole_input_retry(self):
        component = baseline_fact()
        component["atoms"][3]["pos"] = 3

        result = validate([component])

        assert result.components == []
        assert result.retry_needed is True

    def test_self_reference_flags_whole_input_retry(self):
        component = baseline_fact()
        component["atoms"][0]["target_occ"] = 1

        result = validate([component])

        assert result.components == []
        assert result.retry_needed is True

    def test_unknown_target_flags_whole_input_retry(self):
        component = baseline_fact()
        component["atoms"][3]["target_occ"] = 99

        result = validate([component])

        assert result.components == []
        assert result.retry_needed is True

    def test_agent_target_ambiguous_flags_retry(self):
        """小明 appears as agent at two tree levels: no unique repair."""
        component = fact(
            atoms=[
                atom(1, "小明", "E", "agent", 2),
                atom(2, "让", "P", "predicate", None),
                atom(3, "小明", "E", "patient", 2),
                atom(4, "打", "P", "predicate", 3),
                atom(5, "酱油", "E", "patient", 4),
            ],
            tree_=tree(
                "让",
                agent=[arg("小明")],
                patient=[arg("小明")],
                nested=[tree("打", agent=[arg("小明")], patient=[arg("酱油")])],
            ),
        )
        # agent 小明 points at the patient instead of its predicate — and the
        # same text is an agent of both levels, so no unique tree repair.
        component["atoms"][0]["target_occ"] = 3

        result = validate([component])

        assert result.components == []
        assert result.retry_needed is True

    def test_subordinate_predicate_never_demoted(self):
        """A subordinate-clause predicate competing for the root keeps its
        predicate role; only its parent target is repaired."""
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "让", "P", "predicate", None),
                atom(3, "小明", "E", "patient", 2),
                atom(4, "打", "P", "predicate", None),
                atom(5, "酱油", "E", "patient", 4),
            ],
            tree_=tree(
                "让",
                agent=[arg("妈妈")],
                patient=[arg("小明")],
                nested=[tree("打", patient=[arg("酱油")])],
            ),
        )

        result = validate([component])

        assert len(result.components) == 1
        atoms = result.components[0].model_dump()["atoms"]
        assert atoms[3]["role"] == "predicate"
        assert atoms[3]["target_occ"] == 2
        assert result.retry_needed is False

    def test_valid_component_has_no_retry_flag(self):
        result = validate([causative_fact()])

        assert len(result.components) == 1
        assert result.alerts == []
        assert result.retry_needed is False
