"""Frozen Noesis Stage 1 output models for the authoritative event contract."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, StrictBool


class NoesisAtom(BaseModel):
    """One atom occurrence inside a single event closure."""

    model_config = ConfigDict(extra="forbid")

    pos: int = Field(ge=1)
    text: str = Field(min_length=1)
    type: Literal["E", "P"]
    role: Literal["agent", "predicate", "patient", "modifier"]
    target_occ: int | None
    resolved: StrictBool | None


class TreeArgument(BaseModel):
    """An agent/patient argument of a semantic tree node."""

    model_config = ConfigDict(extra="forbid")

    text: str
    modifier: list[str]
    implied: bool


class ConditionalBranch(BaseModel):
    """A conditional branch attached to a semantic tree."""

    model_config = ConfigDict(extra="forbid")

    marker: str | None
    event: "SemanticTree"


class SemanticTree(BaseModel):
    """Recursive human-readable track of one event closure."""

    model_config = ConfigDict(extra="forbid")

    predicate: str
    agent: list[TreeArgument]
    patient: list[TreeArgument]
    modifier: list[str]
    nested: list["SemanticTree"]
    conditional: list[ConditionalBranch]


SemanticTree.model_rebuild()
ConditionalBranch.model_rebuild()


class RulePremise(BaseModel):
    """One premise entry of a hypothesis rule template."""

    model_config = ConfigDict(extra="forbid")

    text: str
    type: Literal["E", "P"]
    role: Literal["agent", "predicate", "patient", "modifier"]


class RuleConclusion(BaseModel):
    """Conclusion of a hypothesis rule template."""

    model_config = ConfigDict(extra="forbid")

    predicate: str
    agent: list[str]
    patient: list[str]
    modifier: list[str]


class RuleTemplate(BaseModel):
    """Structured projection of a hypothesis component."""

    model_config = ConfigDict(extra="forbid")

    premise: list[RulePremise] = Field(min_length=1)
    conclusion: RuleConclusion
    condition: list[str]


class FactComponent(BaseModel):
    """A concrete event closure; rule_template is forbidden."""

    model_config = ConfigDict(extra="forbid")

    utterance_type: Literal["fact"]
    atoms: list[NoesisAtom] = Field(min_length=1)
    tree: SemanticTree


class HypothesisComponent(BaseModel):
    """A regularity candidate; rule_template is required."""

    model_config = ConfigDict(extra="forbid")

    utterance_type: Literal["hypothesis"]
    atoms: list[NoesisAtom] = Field(min_length=1)
    tree: SemanticTree
    rule_template: RuleTemplate


class NoesisExtraction(
    RootModel[
        list[
            Annotated[
                FactComponent | HypothesisComponent,
                Field(discriminator="utterance_type"),
            ]
        ]
    ]
):
    """Authoritative extraction output: a JSON root array of components."""


class ExtractionAlert(BaseModel):
    """Alert object mappable to noesis_core.ingestion_alerts."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal["hyper_extract"]
    alert_code: str
    severity: Literal["info", "warning", "error"]
    message: str
    details: dict[str, Any]


class ValidationResult(BaseModel):
    """Result of deterministic component validation."""

    model_config = ConfigDict(extra="forbid")

    components: list[FactComponent | HypothesisComponent]
    alerts: list[ExtractionAlert]


class ExtractionOutcome(BaseModel):
    """Final extraction outcome including attempt count."""

    model_config = ConfigDict(extra="forbid")

    components: list[FactComponent | HypothesisComponent]
    alerts: list[ExtractionAlert]
    attempts: int
