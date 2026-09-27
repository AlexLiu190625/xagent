"""Tests for the form-answer-continuation model gate.

``form_answer_continuation_enabled`` decides, once per LLM call, whether the
form-answer-continuation text (see ``ExecutionContext.get_messages_for_llm``)
may render for a given LLM. The gate is an exact, case-insensitive match
against an operator-set list (``get_form_answer_continuation_models``); no
model name lives in this repo, and the default (unset env) list is empty.
"""

from __future__ import annotations

import pytest

from xagent.config import FORM_ANSWER_CONTINUATION_MODELS
from xagent.core.agent.pattern.react.react import form_answer_continuation_enabled


class _PlainLLM:
    """An LLM with no ``concrete_model_name`` attribute at all.

    This is the shape of a directly-configured (non-router) BaseLLM, which
    only implements the abstract ``model_name`` property.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


class _RoutedLLM:
    """An LLM exposing both the routing id and the downstream's own name."""

    def __init__(self, *, model_name: str, concrete_model_name: str | None) -> None:
        self.model_name = model_name
        self.concrete_model_name = concrete_model_name


@pytest.mark.parametrize(
    ("env_value", "llm", "expected"),
    [
        pytest.param("some/model", None, False, id="llm-none"),
        pytest.param(None, _PlainLLM("some/model"), False, id="unset-env"),
        pytest.param("some/model", _PlainLLM("some/model"), True, id="exact-match"),
        pytest.param("Some/Model", _PlainLLM("some/MODEL"), True, id="case-different"),
        pytest.param(
            "some/model",
            _PlainLLM("some/model-extended"),
            False,
            id="substring-suffix",
        ),
        pytest.param(
            "some/model",
            _PlainLLM("prefix-some/model"),
            False,
            id="substring-prefix",
        ),
        pytest.param(
            "some/model", _PlainLLM("some/mode"), False, id="substring-shorter"
        ),
        pytest.param("some/model", _PlainLLM(""), False, id="empty-name"),
        pytest.param(
            "concrete/name",
            _RoutedLLM(model_name="auto", concrete_model_name="concrete/name"),
            True,
            # The routing id ("auto") is never what an operator lists; the
            # downstream client's own concrete name is.
            id="routed-concrete-name-listed",
        ),
        pytest.param(
            "auto",
            _RoutedLLM(model_name="auto", concrete_model_name="concrete/name"),
            False,
            # The routing id happens to be listed, but the concrete name
            # (present and non-empty) takes priority and is not listed --
            # must stay disabled, not fall back to the routing id.
            id="routed-only-routing-id-listed",
        ),
        pytest.param(
            "plain/model",
            _RoutedLLM(model_name="plain/model", concrete_model_name=None),
            True,
            id="concrete-name-none-falls-back-to-model-name",
        ),
        pytest.param(
            "plain/model",
            _RoutedLLM(model_name="plain/model", concrete_model_name=""),
            True,
            id="concrete-name-empty-falls-back-to-model-name",
        ),
    ],
)
def test_gate(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    llm: object | None,
    expected: bool,
) -> None:
    if env_value is None:
        monkeypatch.delenv(FORM_ANSWER_CONTINUATION_MODELS, raising=False)
    else:
        monkeypatch.setenv(FORM_ANSWER_CONTINUATION_MODELS, env_value)
    assert form_answer_continuation_enabled(llm) is expected
