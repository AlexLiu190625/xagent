"""Shared anti-fabrication rules for prompts that emit user-facing answers.

Every prompt that produces a final user-facing answer should carry this rule.
That is a design goal, not an enforced invariant: nothing walks the prompt
builders to verify it, and ReAct's no-tool branch still lacks it.

This is the proposal-A mitigation from issue #1235. It reduces how often
unsourced figures are emitted and makes disclosure the instructed default, but
it cannot repair a session whose evidence compaction already discarded.
Proposals B (evidence-preserving compaction) and C (provenance tracking and a
data-source gate) remain open.
"""

from __future__ import annotations


def grounding_rule(*, can_call_tools: bool = True) -> str:
    """Return the grounding rule for user-facing answer generation.

    Args:
        can_call_tools: Whether the receiving LLM call may invoke work tools.
            ``False`` at the three forced-answer sites -- ReAct's forced final
            answer, the DAG completion assessment, and the Auto routing
            decision -- where the rule must tell the model to state the gap
            instead of suggesting a tool call it cannot make.

    Returns:
        A prompt fragment forbidding unsupported claims and unsourced
        quantitative data, and requiring up-front disclosure of any
        illustrative figures.
    """
    insufficient_context_rule = (
        "If available context is insufficient, say so or use an appropriate "
        "tool to verify. "
        if can_call_tools
        else (
            "If available context is insufficient, say so instead of filling "
            "the gap with invented values. "
        )
    )
    return (
        "Do not introduce specific entities, incidents, dates, sources, "
        "causal explanations, or quantitative data (metrics, figures, "
        "statistics, percentages, table rows, or time series) that are not "
        "supported by the conversation, retrieved context, or tool results. "
        f"{insufficient_context_rule}"
        "Never invent figures to fill a gap, and never present invented numbers "
        "as real data; produce unsupported figures only when the user explicitly "
        "asked for a template, mockup, or illustrative example. Labeling is "
        "required either way: if the answer ends up containing any figure that no "
        "tool result or provided context supports, whether or not the user asked "
        "for one, say so up front, before presenting it, and state that those "
        "figures are illustrative placeholders not drawn from any data source."
    )
