"""Unit tests for the frozen Noesis Stage 1 output models (requirement 15.1)."""

import copy

import pytest
from hyperextract.noesis import (
    ConditionalBranch,
    ExtractionAlert,
    ExtractionOutcome,
    FactComponent,
    HypothesisComponent,
    NoesisAtom,
    NoesisExtraction,
    RuleConclusion,
    RulePremise,
    RuleTemplate,
    SemanticTree,
    TreeArgument,
    ValidationResult,
)
from pydantic import ValidationError

MINIMAL_TREE = {
    "predicate": "买",
    "agent": [],
    "patient": [],
    "modifier": [],
    "nested": [],
    "conditional": [],
}

VALID_ATOM = {
    "pos": 1,
    "text": "妈妈",
    "type": "E",
    "role": "agent",
    "target_occ": 2,
    "resolved": None,
}

VALID_FACT_ATOMS = [
    {"pos": 1, "text": "昨天", "type": "E", "role": "modifier", "target_occ": 4, "resolved": None},
    {"pos": 2, "text": "妈妈", "type": "E", "role": "agent", "target_occ": 4, "resolved": None},
    {"pos": 3, "text": "在超市", "type": "E", "role": "modifier", "target_occ": 4, "resolved": None},
    {"pos": 4, "text": "买", "type": "P", "role": "predicate", "target_occ": None, "resolved": None},
    {"pos": 5, "text": "苹果", "type": "E", "role": "patient", "target_occ": 4, "resolved": None},
]

VALID_FACT_TREE = {
    "predicate": "买",
    "agent": [{"text": "妈妈", "modifier": [], "implied": False}],
    "patient": [{"text": "苹果", "modifier": [], "implied": False}],
    "modifier": ["昨天", "在超市"],
    "nested": [],
    "conditional": [],
}

VALID_FACT = {
    "utterance_type": "fact",
    "atoms": VALID_FACT_ATOMS,
    "tree": VALID_FACT_TREE,
}

VALID_HYPOTHESIS_ATOMS = [
    {"pos": 1, "text": "太阳", "type": "E", "role": "agent", "target_occ": 3, "resolved": None},
    {"pos": 2, "text": "每天", "type": "E", "role": "modifier", "target_occ": 3, "resolved": None},
    {"pos": 3, "text": "升起", "type": "P", "role": "predicate", "target_occ": None, "resolved": None},
    {"pos": 4, "text": "东边", "type": "E", "role": "modifier", "target_occ": 3, "resolved": None},
]

VALID_HYPOTHESIS_TREE = {
    "predicate": "升起",
    "agent": [{"text": "太阳", "modifier": [], "implied": False}],
    "patient": [],
    "modifier": ["每天", "东边"],
    "nested": [],
    "conditional": [],
}

VALID_RULE_TEMPLATE = {
    "premise": [{"text": "太阳", "type": "E", "role": "agent"}],
    "conclusion": {
        "predicate": "升起",
        "agent": ["太阳"],
        "patient": [],
        "modifier": ["东边"],
    },
    "condition": ["每天"],
}

VALID_HYPOTHESIS = {
    "utterance_type": "hypothesis",
    "atoms": VALID_HYPOTHESIS_ATOMS,
    "tree": VALID_HYPOTHESIS_TREE,
    "rule_template": VALID_RULE_TEMPLATE,
}


class TestRootOutput:
    """The authoritative output is a JSON root array, never a wrapper object."""

    def test_empty_array_is_valid(self):
        parsed = NoesisExtraction.model_validate([])

        assert parsed.root == []

    def test_wrapped_components_object_rejected(self):
        with pytest.raises(ValidationError):
            NoesisExtraction.model_validate({"components": [VALID_FACT]})

    def test_single_component_object_rejected(self):
        with pytest.raises(ValidationError):
            NoesisExtraction.model_validate(VALID_FACT)

    def test_non_array_rejected(self):
        with pytest.raises(ValidationError):
            NoesisExtraction.model_validate("昨天妈妈在超市买了苹果。")


class TestDiscriminatedUnion:
    """fact/hypothesis dispatch on the utterance_type discriminator."""

    def test_fact_dispatched(self):
        parsed = NoesisExtraction.model_validate([VALID_FACT])

        assert len(parsed.root) == 1
        assert isinstance(parsed.root[0], FactComponent)

    def test_hypothesis_dispatched(self):
        parsed = NoesisExtraction.model_validate([VALID_HYPOTHESIS])

        assert len(parsed.root) == 1
        assert isinstance(parsed.root[0], HypothesisComponent)

    def test_mixed_components_preserve_order(self):
        parsed = NoesisExtraction.model_validate([VALID_FACT, VALID_HYPOTHESIS])

        assert isinstance(parsed.root[0], FactComponent)
        assert isinstance(parsed.root[1], HypothesisComponent)

    def test_unknown_utterance_type_rejected(self):
        noise = copy.deepcopy(VALID_FACT)
        noise["utterance_type"] = "law"

        with pytest.raises(ValidationError):
            NoesisExtraction.model_validate([noise])


class TestRuleTemplatePresence:
    """fact forbids rule_template; hypothesis requires it."""

    def test_fact_with_rule_template_rejected(self):
        fact = copy.deepcopy(VALID_FACT)
        fact["rule_template"] = copy.deepcopy(VALID_RULE_TEMPLATE)

        with pytest.raises(ValidationError):
            FactComponent.model_validate(fact)

    def test_hypothesis_without_rule_template_rejected(self):
        hypothesis = copy.deepcopy(VALID_HYPOTHESIS)
        del hypothesis["rule_template"]

        with pytest.raises(ValidationError):
            HypothesisComponent.model_validate(hypothesis)

    def test_hypothesis_with_null_rule_template_rejected(self):
        hypothesis = copy.deepcopy(VALID_HYPOTHESIS)
        hypothesis["rule_template"] = None

        with pytest.raises(ValidationError):
            HypothesisComponent.model_validate(hypothesis)


class TestComponentAtoms:
    """A non-empty component must contain at least one atom occurrence."""

    @pytest.mark.parametrize("component", [VALID_FACT, VALID_HYPOTHESIS])
    def test_empty_atoms_rejected_by_schema(self, component):
        payload = copy.deepcopy(component)
        payload["atoms"] = []

        with pytest.raises(ValidationError):
            NoesisExtraction.model_validate([payload])

EXTRA_FORBID_CASES = [
    (
        NoesisAtom,
        VALID_ATOM,
    ),
    (
        TreeArgument,
        {"text": "妈妈", "modifier": [], "implied": False},
    ),
    (
        SemanticTree,
        MINIMAL_TREE,
    ),
    (
        ConditionalBranch,
        {"marker": None, "event": MINIMAL_TREE},
    ),
    (
        RulePremise,
        {"text": "太阳", "type": "E", "role": "agent"},
    ),
    (
        RuleConclusion,
        {"predicate": "升起", "agent": [], "patient": [], "modifier": []},
    ),
    (
        RuleTemplate,
        copy.deepcopy(VALID_RULE_TEMPLATE),
    ),
    (
        FactComponent,
        VALID_FACT,
    ),
    (
        HypothesisComponent,
        VALID_HYPOTHESIS,
    ),
    (
        ExtractionAlert,
        {
            "stage": "hyper_extract",
            "alert_code": "invalid_component_dropped",
            "severity": "warning",
            "message": "dropped",
            "details": {},
        },
    ),
    (
        ValidationResult,
        {"components": [], "alerts": []},
    ),
    (
        ExtractionOutcome,
        {"components": [], "alerts": [], "attempts": 1},
    ),
]

EXTRA_FORBID_IDS = [case[0].__name__ for case in EXTRA_FORBID_CASES]


class TestExtraForbidden:
    """Every nested model is strict: unknown fields are rejected."""

    @pytest.mark.parametrize(("model_cls", "payload"), EXTRA_FORBID_CASES, ids=EXTRA_FORBID_IDS)
    def test_baseline_payload_is_schema_valid(self, model_cls, payload):
        model_cls.model_validate(copy.deepcopy(payload))

    @pytest.mark.parametrize(("model_cls", "payload"), EXTRA_FORBID_CASES, ids=EXTRA_FORBID_IDS)
    def test_unknown_field_rejected(self, model_cls, payload):
        payload = copy.deepcopy(payload)
        payload["unexpected_field"] = 1

        with pytest.raises(ValidationError):
            model_cls.model_validate(payload)


class TestAtomSchema:
    """NoesisAtom has exactly six required fields with strict enums."""

    @pytest.mark.parametrize(
        "missing_field", ["pos", "text", "type", "role", "target_occ", "resolved"]
    )
    def test_missing_required_field_rejected(self, missing_field):
        atom = copy.deepcopy(VALID_ATOM)
        del atom[missing_field]

        with pytest.raises(ValidationError):
            NoesisAtom.model_validate(atom)

    @pytest.mark.parametrize("bad_pos", [0, -1])
    def test_pos_must_be_at_least_one(self, bad_pos):
        atom = copy.deepcopy(VALID_ATOM)
        atom["pos"] = bad_pos

        with pytest.raises(ValidationError):
            NoesisAtom.model_validate(atom)

    def test_empty_text_rejected(self):
        atom = copy.deepcopy(VALID_ATOM)
        atom["text"] = ""

        with pytest.raises(ValidationError):
            NoesisAtom.model_validate(atom)

    @pytest.mark.parametrize("bad_type", ["C", "X", "e"])
    def test_type_enum_strict(self, bad_type):
        atom = copy.deepcopy(VALID_ATOM)
        atom["type"] = bad_type

        with pytest.raises(ValidationError):
            NoesisAtom.model_validate(atom)

    @pytest.mark.parametrize("bad_role", ["subject", "object", "root", "AGENT"])
    def test_role_enum_strict(self, bad_role):
        atom = copy.deepcopy(VALID_ATOM)
        atom["role"] = bad_role

        with pytest.raises(ValidationError):
            NoesisAtom.model_validate(atom)

    @pytest.mark.parametrize("role", ["agent", "predicate", "patient", "modifier"])
    def test_role_accepts_all_four_values(self, role):
        atom = copy.deepcopy(VALID_ATOM)
        atom["role"] = role

        assert NoesisAtom.model_validate(atom).role == role

    @pytest.mark.parametrize("resolved", [True, False, None])
    def test_resolved_accepts_bool_or_null(self, resolved):
        atom = copy.deepcopy(VALID_ATOM)
        atom["resolved"] = resolved

        assert NoesisAtom.model_validate(atom).resolved is resolved

    @pytest.mark.parametrize("bad_resolved", ["true", "false", "null", 1, 0])
    def test_resolved_rejects_non_bool_values(self, bad_resolved):
        atom = copy.deepcopy(VALID_ATOM)
        atom["resolved"] = bad_resolved

        with pytest.raises(ValidationError):
            NoesisAtom.model_validate(atom)


class TestTreeArrays:
    """SemanticTree array fields never accept objects or scalars."""

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("agent", {"text": "妈妈", "modifier": [], "implied": False}),
            ("agent", "妈妈"),
            ("patient", "苹果"),
            ("modifier", "昨天"),
            ("nested", MINIMAL_TREE),
            ("conditional", "如果"),
        ],
    )
    def test_array_field_rejects_object_or_scalar(self, field, bad_value):
        tree = copy.deepcopy(MINIMAL_TREE)
        tree[field] = bad_value

        with pytest.raises(ValidationError):
            SemanticTree.model_validate(tree)

    @pytest.mark.parametrize(
        "missing_field", ["predicate", "agent", "patient", "modifier", "nested", "conditional"]
    )
    def test_missing_tree_field_rejected(self, missing_field):
        tree = copy.deepcopy(MINIMAL_TREE)
        del tree[missing_field]

        with pytest.raises(ValidationError):
            SemanticTree.model_validate(tree)

    def test_tree_is_recursive(self):
        tree = copy.deepcopy(MINIMAL_TREE)
        tree["nested"] = [copy.deepcopy(MINIMAL_TREE)]

        parsed = SemanticTree.model_validate(tree)

        assert isinstance(parsed.nested[0], SemanticTree)


class TestRuleTemplateSchema:
    """rule_template fields are all required; premise needs at least one entry."""

    def test_empty_premise_rejected(self):
        template = copy.deepcopy(VALID_RULE_TEMPLATE)
        template["premise"] = []

        with pytest.raises(ValidationError):
            RuleTemplate.model_validate(template)

    @pytest.mark.parametrize("missing_field", ["premise", "conclusion", "condition"])
    def test_missing_field_rejected(self, missing_field):
        template = copy.deepcopy(VALID_RULE_TEMPLATE)
        del template[missing_field]

        with pytest.raises(ValidationError):
            RuleTemplate.model_validate(template)

    def test_empty_condition_allowed(self):
        template = copy.deepcopy(VALID_RULE_TEMPLATE)
        template["condition"] = []

        assert RuleTemplate.model_validate(template).condition == []

    @pytest.mark.parametrize("missing_field", ["predicate", "agent", "patient", "modifier"])
    def test_conclusion_missing_field_rejected(self, missing_field):
        template = copy.deepcopy(VALID_RULE_TEMPLATE)
        del template["conclusion"][missing_field]

        with pytest.raises(ValidationError):
            RuleTemplate.model_validate(template)


class TestAlertSchema:
    """ExtractionAlert enums are frozen to stage/severity literals."""

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValidationError):
            ExtractionAlert.model_validate(
                {
                    "stage": "hyper_extract",
                    "alert_code": "extraction_failed",
                    "severity": "critical",
                    "message": "failed",
                    "details": {},
                }
            )

    def test_invalid_stage_rejected(self):
        with pytest.raises(ValidationError):
            ExtractionAlert.model_validate(
                {
                    "stage": "hindsight",
                    "alert_code": "extraction_failed",
                    "severity": "error",
                    "message": "failed",
                    "details": {},
                }
            )
