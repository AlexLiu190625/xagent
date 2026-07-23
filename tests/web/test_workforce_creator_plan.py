"""Unit tests for the Workforce creation-plan shape (#800).

Workforce-level manager instructions were removed: the cleaned plan must not
carry a top-level ``manager_instructions`` key, while the manager spec itself
keeps its instructions.
"""

from xagent.web.services.workforce_creator import (
    _clean_creation_plan,
    _fallback_creation_plan,
)


def test_clean_creation_plan_has_no_top_level_manager_instructions() -> None:
    plan = _clean_creation_plan(
        {
            "name": "Research Workforce",
            "description": "Coordinates research",
            "manager": {
                "name": "Research Manager",
                "description": "Coordinates workers",
                "instructions": "Delegate and synthesize.",
            },
            "manager_instructions": "Legacy top-level value",
            "workers": [],
        },
        available_agent_ids=set(),
        prompt="research",
    )

    assert "manager_instructions" not in plan
    assert plan["manager"]["instructions"] == "Delegate and synthesize."


def test_fallback_creation_plan_has_no_top_level_manager_instructions() -> None:
    plan = _fallback_creation_plan("research assistant", agents=[])

    assert "manager_instructions" not in plan
    assert plan["manager"]["instructions"]
