"""Closure edge rules, type/role combinations and tree accounting."""

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


def branch(event, marker=None):
    return {"marker": marker, "event": event}


def validate(raw, source_text):
    return validate_components(raw, source_text=source_text)


def assert_dropped(result):
    assert result.components == []
    assert len(result.alerts) == 1
    assert result.alerts[0].alert_code == "invalid_component_dropped"
    assert result.alerts[0].severity == "warning"
    assert result.alerts[0].stage == "hyper_extract"


class TestClosureEdgeRules:
    """Section 6.2: agent/patient point at their SPO predicate."""

    def test_agent_pointing_to_patient_dropped(self):
        component = fact(
            atoms=[
                atom(1, "买", "P", "predicate", None),
                atom(2, "妈妈", "E", "agent", 3),
                atom(3, "苹果", "E", "patient", 1),
            ],
            tree_=tree("买", agent=[arg("妈妈")], patient=[arg("苹果")]),
        )

        result = validate([component], "妈妈买苹果")

        assert_dropped(result)

    def test_patient_pointing_to_agent_dropped(self):
        component = fact(
            atoms=[
                atom(1, "买", "P", "predicate", None),
                atom(2, "妈妈", "E", "agent", 1),
                atom(3, "苹果", "E", "patient", 2),
            ],
            tree_=tree("买", agent=[arg("妈妈")], patient=[arg("苹果")]),
        )

        result = validate([component], "妈妈买苹果")

        assert_dropped(result)

    def test_modifier_pointing_to_modifier_dropped(self):
        component = fact(
            atoms=[
                atom(1, "买", "P", "predicate", None),
                atom(2, "妈妈", "E", "agent", 1),
                atom(3, "昨天", "E", "modifier", 4),
                atom(4, "在超市", "E", "modifier", 1),
            ],
            tree_=tree("买", agent=[arg("妈妈")], modifier=["昨天", "在超市"]),
        )

        result = validate([component], "昨天妈妈在超市买了苹果")

        assert_dropped(result)

    def test_sentence_modifier_predicate_can_point_to_argument(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "P", "predicate", 3),
                atom(3, "苹果", "E", "agent", 4),
                atom(4, "甜", "P", "predicate", None),
            ],
            tree_=tree(
                "甜",
                agent=[arg("苹果")],
                nested=[
                    tree(
                        "买",
                        agent=[arg("妈妈")],
                        patient=[arg("苹果", implied=True)],
                    )
                ],
            ),
        )

        result = validate([component], "妈妈买的苹果很甜")

        assert len(result.components) == 1
        assert result.alerts == []

    def test_subordinate_predicate_pointing_to_modifier_dropped(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "让", "P", "predicate", None),
                atom(3, "快速", "E", "modifier", 2),
                atom(4, "打", "P", "predicate", 3),
                atom(5, "酱油", "E", "patient", 4),
            ],
            tree_=tree(
                "让",
                agent=[arg("妈妈")],
                modifier=["快速"],
                nested=[tree("打", patient=[arg("酱油")])],
            ),
        )

        result = validate([component], "妈妈让快速打酱油")

        assert_dropped(result)


class TestAtomTypeRoleCombinations:
    """Predicates are P; E/G may fill entity-like roles; modifiers may use any type."""

    def test_predicate_atom_with_type_e_dropped(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "E", "predicate", None),
                atom(3, "苹果", "E", "patient", 2),
            ],
            tree_=tree("买", agent=[arg("妈妈")], patient=[arg("苹果")]),
        )

        result = validate([component], "妈妈买苹果")

        assert_dropped(result)

    def test_agent_atom_with_type_p_dropped(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "P", "agent", 2),
                atom(2, "买", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 2),
            ],
            tree_=tree("买", agent=[arg("妈妈")], patient=[arg("苹果")]),
        )

        result = validate([component], "妈妈买苹果")

        assert_dropped(result)

    def test_patient_atom_with_type_p_dropped(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "P", "predicate", None),
                atom(3, "苹果", "P", "patient", 2),
            ],
            tree_=tree("买", agent=[arg("妈妈")], patient=[arg("苹果")]),
        )

        result = validate([component], "妈妈买苹果")

        assert_dropped(result)

    def test_modifier_atom_with_type_p_passes(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 2),
                atom(4, "没带钱", "P", "modifier", 2),
            ],
            tree_=tree(
                "买", agent=[arg("妈妈")], patient=[arg("苹果")], modifier=["没带钱"]
            ),
        )

        result = validate([component], "妈妈没带钱买了苹果")

        assert len(result.components) == 1
        assert result.alerts == []

    def test_geneme_patient_passes_as_constructed_concept(self):
        component = fact(
            atoms=[
                atom(1, "系统", "E", "agent", 2),
                atom(2, "识别", "P", "predicate", None),
                atom(3, "生长周期", "G", "patient", 2),
            ],
            tree_=tree("识别", agent=[arg("系统")], patient=[arg("生长周期")]),
        )

        result = validate([component], "系统识别生长周期")

        assert len(result.components) == 1
        assert result.alerts == []

    def test_geneme_can_be_inherited_as_implied_nested_argument(self):
        component = fact(
            atoms=[
                atom(1, "生长周期", "G", "patient", 2),
                atom(2, "形成", "P", "predicate", None),
                atom(3, "结束", "P", "predicate", 2),
            ],
            tree_=tree(
                "形成",
                patient=[arg("生长周期")],
                nested=[tree("结束", patient=[arg("生长周期", implied=True)])],
            ),
        )

        result = validate([component], "生长周期形成后结束")

        assert len(result.components) == 1
        assert result.alerts == []


class TestTreeStructuralAccounting:
    """Section 7.4: tree must mirror atoms by role, target and occurrence."""

    def test_tree_arg_role_mismatch_dropped(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 2),
            ],
            tree_=tree("买", patient=[arg("妈妈"), arg("苹果")]),
        )

        result = validate([component], "妈妈买苹果")

        assert_dropped(result)

    def test_patient_modifier_attached_to_root_dropped(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 2),
                atom(4, "新鲜的", "E", "modifier", 3),
            ],
            tree_=tree(
                "买", agent=[arg("妈妈")], patient=[arg("苹果")], modifier=["新鲜的"]
            ),
        )

        result = validate([component], "妈妈买了新鲜的苹果")

        assert_dropped(result)

    def test_subordinate_predicate_missing_from_tree_dropped(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "让", "P", "predicate", None),
                atom(3, "小明", "E", "patient", 2),
                atom(4, "打", "P", "predicate", 2),
                atom(5, "酱油", "E", "patient", 4),
            ],
            tree_=tree("让", agent=[arg("妈妈")], patient=[arg("小明")]),
        )

        result = validate([component], "妈妈让小明打酱油")

        assert_dropped(result)

    def test_duplicate_tree_agent_dropped(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 2),
            ],
            tree_=tree("买", agent=[arg("妈妈"), arg("妈妈")], patient=[arg("苹果")]),
        )

        result = validate([component], "妈妈买苹果")

        assert_dropped(result)

    def test_wrong_occurrence_same_literal_dropped(self):
        component = fact(
            atoms=[
                atom(1, "小明", "E", "agent", 4),
                atom(2, "没写", "P", "predicate", 4),
                atom(3, "作业", "E", "patient", 2),
                atom(4, "揍", "P", "predicate", None),
            ],
            tree_=tree(
                "揍",
                agent=[arg("小明")],
                nested=[tree("没写", agent=[arg("小明")], patient=[arg("作业")])],
            ),
        )

        result = validate([component], "小明没写作业后揍了自己")

        assert_dropped(result)

    def test_agent_atom_missing_from_tree_dropped(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "买", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 2),
            ],
            tree_=tree("买", patient=[arg("苹果")]),
        )

        result = validate([component], "妈妈买苹果")

        assert_dropped(result)

    def test_modifier_atom_missing_from_tree_dropped(self):
        component = fact(
            atoms=[
                atom(1, "昨天", "E", "modifier", 2),
                atom(2, "买", "P", "predicate", None),
                atom(3, "苹果", "E", "patient", 2),
            ],
            tree_=tree("买", patient=[arg("苹果")]),
        )

        result = validate([component], "昨天买了苹果")

        assert_dropped(result)

    def test_nested_predicate_without_atom_dropped(self):
        component = fact(
            atoms=[
                atom(1, "妈妈", "E", "agent", 2),
                atom(2, "让", "P", "predicate", None),
                atom(3, "小明", "E", "patient", 2),
            ],
            tree_=tree(
                "让",
                agent=[arg("妈妈")],
                patient=[arg("小明")],
                nested=[tree("打", patient=[arg("酱油")])],
            ),
        )

        result = validate([component], "妈妈让小明打酱油")

        assert_dropped(result)

    def test_conditional_marker_not_modifier_atom_dropped(self):
        component = fact(
            atoms=[
                atom(1, "如果", "E", "patient", 2),
                atom(2, "带", "P", "predicate", None),
                atom(3, "下雨", "P", "predicate", 2),
                atom(4, "小明", "E", "agent", 2),
            ],
            tree_=tree(
                "带",
                agent=[arg("小明")],
                patient=[arg("如果")],
                conditional=[branch(tree("下雨"), marker="如果")],
            ),
        )

        result = validate([component], "如果下雨小明带伞")

        assert_dropped(result)

    def test_multilevel_valid_structure_passes(self):
        component = fact(
            atoms=[
                atom(1, "小明", "E", "agent", 6),
                atom(2, "如果", "E", "modifier", 6),
                atom(3, "明天", "E", "modifier", 4),
                atom(4, "下雨", "P", "predicate", 6),
                atom(5, "带", "P", "predicate", 6),
                atom(6, "让", "P", "predicate", None),
                atom(7, "妈妈", "E", "patient", 6),
                atom(8, "伞", "E", "patient", 5),
            ],
            tree_=tree(
                "让",
                agent=[arg("小明")],
                patient=[arg("妈妈")],
                nested=[tree("带", agent=[arg("妈妈", implied=True)], patient=[arg("伞")])],
                conditional=[branch(tree("下雨", modifier=["明天"]), marker="如果")],
            ),
        )

        result = validate([component], "如果明天下雨小明让妈妈带伞")

        assert len(result.components) == 1
        assert result.alerts == []
