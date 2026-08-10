from __future__ import annotations

from xagent.core.agent.grounding import grounding_rule


def test_grounding_rule_covers_quantitative_data() -> None:
    rule = grounding_rule()

    for term in (
        "entities",
        "incidents",
        "dates",
        "sources",
        "causal explanations",
        "quantitative data",
        "metrics",
        "figures",
        "statistics",
        "percentages",
        "table rows",
        "time series",
    ):
        assert term in rule
    assert "use an appropriate tool to verify" in rule
    assert "illustrative placeholders" in rule
    # Pin the concatenation: a dropped trailing space would still satisfy
    # every membership assertion above.
    assert "verify. Never invent figures" in rule


def test_grounding_rule_without_tools_omits_tool_verification() -> None:
    rule = grounding_rule(can_call_tools=False)

    assert "use an appropriate tool" not in rule
    assert "invented values" in rule
    assert "quantitative data" in rule
    assert "illustrative placeholders" in rule
    assert "invented values. Never invent figures" in rule


def test_grounding_rule_requires_labeling_regardless_of_request() -> None:
    """Disclosure must not be conditional on the user asking for a template.

    The #1235 session asked for a real KPI report, so a disclosure duty gated
    on "user asked for an example" would not have applied to it.
    """
    for rule in (grounding_rule(), grounding_rule(can_call_tools=False)):
        assert "Labeling is required either way" in rule
        assert "whether or not the user asked for one" in rule
        assert "produce unsupported figures only when the user explicitly" in rule
