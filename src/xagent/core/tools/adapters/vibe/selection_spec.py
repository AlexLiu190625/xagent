"""Declarative tool-selection spec for :class:`ToolFactory`.

Background:
    Before this spec, :func:`ToolFactory.create_all_tools` built the
    full set of registered tools and filtered them by name afterwards
    via ``config.get_allowed_tools()``. Callers that only needed a
    category-level filter (e.g. the WS chat path) had to pre-build the
    entire ~52 tool list just to read each tool's metadata.category
    and assemble a name list -- a redundant build that dominated
    per-task setup time (see issue #427).

    ``ToolSelectionSpec`` is a sealed ABC with three concrete
    subclasses (``_SpecAll`` / ``_SpecNone`` / ``_SpecByCategories``).
    Modes are explicit through subclass identity and
    :meth:`is_all` / :meth:`is_none` / :meth:`is_by_categories`
    predicates -- not the older "None vs frozenset() vs frozenset({...})"
    implicit signal that conflated the three states and caused
    legacy ``Agent.tool_categories=[]`` to be misread as "zero tools".

    Production callers MUST construct via
    :meth:`ToolSelectionSpec.from_raw`, the single normalizer over
    raw ORM / dict / SDK fields. Direct subclass instantiation is
    used by tests; production paths that bypass ``from_raw`` are
    flagged by a grep test.

Mode completeness:
    Each abstract method (``is_*`` / ``includes_*`` /
    ``compute_allowed_names``) must be implemented by every subclass.
    Missing an implementation is both a mypy error and a runtime
    ``TypeError`` at instantiation time. Adding a new ``includes_*``
    creator-dispatch method on the base forces every subclass to
    update -- no grep test required to police mode dispatch.

Backward compat:
    All three subclasses expose ``categories`` /``mcp_servers`` /
    ``custom_api_ids`` / ``published_agent_ids`` fields (the original
    spec shape). ``_SpecAll`` has them at None / ``_SpecNone`` at
    None plus empty ``categories``, so existing callsites that read
    ``spec.categories`` directly keep working. New code should
    prefer the typed dispatch (``spec.is_by_categories()`` etc.).

This module deliberately has no dependencies on the rest of the
codebase so the spec can be imported by both the factory and the
individual tool creators without circular-import risk. The
``compute_allowed_names`` helper uses duck typing for tool metadata
access; no Tool type import required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set


class ToolSelectionSpec(ABC):
    """Sealed type for tool selection.

    Three concrete subclasses, accessed through :meth:`from_raw`:
      - ``_SpecAll`` — legacy "未配置" / no restriction; build every
        default tool. Factory does not filter by name.
      - ``_SpecNone`` — explicit "zero tools"; factory returns ``[]``.
      - ``_SpecByCategories`` — filter by category, with optional
        ID-level scopes (mcp_servers / custom_api_ids /
        published_agent_ids) and ``name_extras`` (workforce worker
        tool injection).

    Mode completeness is enforced by ``@abstractmethod``: each
    subclass must implement every predicate / dispatch method.
    Missing one fails at instantiation time, not silently.

    Direct ``ToolSelectionSpec(...)`` construction is supported for
    backward compatibility: ``__new__`` inspects ``categories`` and
    dispatches to the matching subclass. Production code should
    prefer :meth:`from_raw` for the explicit normalizer contract;
    direct construction is mostly for tests + legacy callers.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "ToolSelectionSpec":
        """Dispatch ``ToolSelectionSpec(...)`` to the right subclass.

        Backward-compat shim for callers / tests that construct the
        old single-dataclass shape directly. New production code
        should prefer :meth:`from_raw`.

        Dispatch rules (match the legacy dataclass semantics):
          - ``categories=None`` (default) → ``_SpecAll``
          - ``categories=frozenset()``    → ``_SpecNone``
          - ``categories=frozenset({...})`` non-empty → ``_SpecByCategories``

        Subclass direct construction (``_SpecAll()``, ``_SpecNone()``,
        ``_SpecByCategories(...)``) bypasses this dispatch.
        """
        if cls is ToolSelectionSpec:
            categories = kwargs.get("categories")
            if categories is None:
                return _SpecAll.__new__(_SpecAll)
            if isinstance(categories, frozenset) and len(categories) == 0:
                return _SpecNone.__new__(_SpecNone)
            return _SpecByCategories.__new__(_SpecByCategories)
        return super().__new__(cls)

    # ── Mode predicates ────────────────────────────────────────────
    # Three mutually exclusive modes; exactly one is_*() returns True.

    @abstractmethod
    def is_all(self) -> bool:
        """Whether this is the ALL mode (build every default tool)."""

    @abstractmethod
    def is_none(self) -> bool:
        """Whether this is the NONE mode (factory returns ``[]``)."""

    @abstractmethod
    def is_by_categories(self) -> bool:
        """Whether this is the BY_CATEGORIES mode (filtered build)."""

    # ── Creator dispatch ──────────────────────────────────────────
    # ToolRegistry / individual creators consult these to decide
    # whether their work (DB queries / MCP init / etc) should run.

    @abstractmethod
    def includes_mcp(self) -> bool:
        """Whether the MCP creator should run."""

    @abstractmethod
    def includes_custom_api(self) -> bool:
        """Whether the Custom API creator should run.

        In BY_CATEGORIES mode this also requires ``"other"`` in
        :attr:`categories` because Custom API tools surface under
        the ``other`` category; without it they cannot survive the
        post-build name filter, so running the creator (and its
        ``get_custom_api_configs()`` DB lookup) is wasted I/O.
        """

    @abstractmethod
    def includes_published_agent(self) -> bool:
        """Whether the Published Agent delegation creators should run."""

    # ── Final name-level filter ───────────────────────────────────

    @abstractmethod
    def compute_allowed_names(self, all_tools: List[Any]) -> Optional[frozenset[str]]:
        """Resolve the final allowed-tool-names set for this spec.

        Returns:
            ``None``       — caller keeps every tool in ``all_tools``
                             (ALL mode).
            ``frozenset()`` — caller returns ``[]`` (NONE mode).
            non-empty set  — caller filters ``all_tools`` to names
                             in the set (BY_CATEGORIES mode, plus
                             :attr:`name_extras` injection).

        The frozenset() vs None distinction is load-bearing:
        ``ToolFactory.create_all_tools`` reads this method and
        filters / short-circuits based on the three return types.
        """

    # ── Single normalizer ─────────────────────────────────────────

    @classmethod
    def from_raw(
        cls,
        *,
        tool_categories: Optional[List[str]] = None,
        mcp_servers: Optional[List[str]] = None,
        custom_api_ids: Optional[List[int]] = None,
        published_agent_ids: Optional[List[int]] = None,
        workforce_extra_names: Optional[Set[str]] = None,
        explicit_none: bool = False,
        extras_only_when_unconfigured: bool = False,
    ) -> "ToolSelectionSpec":
        """Build a spec from raw ORM / dict / SDK fields.

        This is the **only** production entry point. Direct
        subclass construction is for tests; a grep test pins
        production code to use ``from_raw``.

        Empty / unset input semantics:
          - ``tool_categories=None`` or ``[]`` → ``_SpecAll``
            (legacy "未配置" — build every default tool).
          - ``explicit_none=True`` → ``_SpecNone`` regardless of
            ``tool_categories`` (reserved for future "zero tools"
            product UI).
          - ``extras_only_when_unconfigured=True`` with unset / empty
            categories → only ``workforce_extra_names`` are admitted
            (or ``_SpecNone`` when there are no extras). Workforce
            manager tasks use this so an unconfigured manager can only
            delegate to its workers by default, without inheriting the
            full ordinary tool set.
          - Otherwise → ``_SpecByCategories``.

        By default, ``workforce_extra_names`` is only meaningful in
        BY_CATEGORIES mode (ALL already includes everything; NONE
        rejects everything). ``extras_only_when_unconfigured`` is the
        opt-in exception for workforce manager runtime construction.
        """
        if explicit_none:
            return _SpecNone()
        if tool_categories is None or len(tool_categories) == 0:
            if extras_only_when_unconfigured:
                name_extras = frozenset(workforce_extra_names or ())
                if not name_extras:
                    return _SpecNone()
                return _SpecByCategories(
                    categories=frozenset(),
                    name_extras=name_extras,
                    _user_picked=frozenset(),
                )
            return _SpecAll()

        # Agent-builder UI representation in ``tool_categories`` mixes
        # two shapes:
        #   - plain category names (``"basic"``, ``"file"``, ``"mcp"``)
        #   - ``"mcp:<server-name>"`` (selects a specific MCP server
        #     OR a Custom-API tool that fronts it; both surface as
        #     name patterns under the ``"mcp"`` / ``"other"`` categories)
        #
        # Normalize the mixed shape into the structured spec form:
        #   - plain entries land in ``categories``
        #   - ``mcp:<server>`` entries add ``"mcp"`` + ``"other"`` to
        #     ``categories`` (so MCP / Custom-API creators run) and
        #     ``<server>`` to ``mcp_servers`` (per-server filter)
        #
        # The ``categories`` frozenset retains the original
        # ``mcp:<server>`` strings too — :meth:`compute_allowed_names`
        # uses them to match tool names (``mcp_<server>_*`` /
        # ``api_<server>_call``) at the name-filter step.
        derived_cats: Set[str] = set()
        derived_mcp_servers: Set[str] = set()
        for entry in tool_categories:
            if isinstance(entry, str) and entry.startswith("mcp:"):
                server_name = entry.split(":", 1)[1].replace(" ", "_").replace("-", "_")
                derived_cats.add("mcp")
                derived_cats.add("other")
                derived_cats.add(entry)  # keep for name-filter step
                derived_mcp_servers.add(server_name)
            else:
                derived_cats.add(entry)

        # Caller-supplied mcp_servers (if any) merge with the
        # derived set; explicit empty stays empty.
        if mcp_servers is not None:
            final_mcp_servers: Optional[frozenset[str]] = frozenset(mcp_servers)
        elif derived_mcp_servers:
            final_mcp_servers = frozenset(derived_mcp_servers)
        else:
            final_mcp_servers = None

        return _SpecByCategories(
            categories=frozenset(derived_cats),
            mcp_servers=final_mcp_servers,
            custom_api_ids=(
                frozenset(custom_api_ids) if custom_api_ids is not None else None
            ),
            published_agent_ids=(
                frozenset(published_agent_ids)
                if published_agent_ids is not None
                else None
            ),
            name_extras=frozenset(workforce_extra_names or ()),
            # Record the user's raw category list so
            # ``compute_allowed_names`` can tell plain "mcp" / "other"
            # admit-all from a derived "mcp" added solely for the
            # registry-skip side of mcp:<server> sub-categories.
            _user_picked=frozenset(tool_categories),
        )

    # ── Backward-compat helper (kept from the original spec) ─────

    def includes_category(self, cat: str) -> bool:
        """Whether the given category passes the spec.

        ``ALL`` admits every category; ``NONE`` admits none;
        ``BY_CATEGORIES`` admits members of :attr:`categories`.
        Existing callers in ``factory.py`` registry-skip and
        creator-internal short-circuits keep using this.
        """
        if self.is_all():
            return True
        if self.is_none():
            return False
        # ``categories`` exists on _SpecByCategories; mypy follows it
        # through ``is_by_categories()`` narrowing in modern setups,
        # but the duck-typed attribute access is also safe here.
        return cat in getattr(self, "categories", frozenset())


# ─────────────────────────────────────────────────────────────────
# Concrete subclasses. Production code MUST go through ``from_raw``.
# The leading underscore signals "internal" — direct construction is
# legal but flagged by a grep test in non-test paths.
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SpecAll(ToolSelectionSpec):
    """ALL mode — legacy "未配置" / no restriction.

    Exposes ``categories`` / ``mcp_servers`` / ``custom_api_ids`` /
    ``published_agent_ids`` at ``None`` for backward compat with
    callsites that read those attributes directly (e.g. the
    registry-level skip in ``factory.py:ToolRegistry``).
    """

    # Backward-compat fields (kept None to preserve existing
    # ``spec.categories is None`` truthiness in factory.py).
    categories: Optional[frozenset[str]] = None
    mcp_servers: Optional[frozenset[str]] = None
    custom_api_ids: Optional[frozenset[int]] = None
    published_agent_ids: Optional[frozenset[int]] = None

    def is_all(self) -> bool:
        return True

    def is_none(self) -> bool:
        return False

    def is_by_categories(self) -> bool:
        return False

    def includes_mcp(self) -> bool:
        # Backward-compat: legacy callers could write
        # ToolSelectionSpec(mcp_servers=frozenset()) to express
        # "no MCP tools" even without a categories filter. Honor
        # that here (the abstract method is still enforced -- this
        # is just the ALL subclass's concrete implementation).
        if self.mcp_servers is not None and len(self.mcp_servers) == 0:
            return False
        return True

    def includes_custom_api(self) -> bool:
        # Backward-compat mirror of includes_mcp above for the
        # explicit-exclude legacy shape.
        if self.custom_api_ids is not None and len(self.custom_api_ids) == 0:
            return False
        return True

    def includes_published_agent(self) -> bool:
        if self.published_agent_ids is not None and len(self.published_agent_ids) == 0:
            return False
        return True

    def compute_allowed_names(self, all_tools: List[Any]) -> Optional[frozenset[str]]:
        # None signals "no name-level filter" -- factory keeps
        # every tool returned by the registry.
        return None


@dataclass(frozen=True)
class _SpecNone(ToolSelectionSpec):
    """NONE mode — explicit "zero tools" (no UI entry today, reserved).

    ``categories`` is an explicit empty frozenset so existing
    callsites that test ``spec.categories is not None`` see "set"
    and walk the "no intersection" branch (factory.py:ToolRegistry
    then skips every creator). This mirrors the original
    ``categories=frozenset()`` "explicit exclusion" semantics
    documented on the old dataclass.
    """

    categories: Optional[frozenset[str]] = field(default_factory=lambda: frozenset())
    mcp_servers: Optional[frozenset[str]] = None
    custom_api_ids: Optional[frozenset[int]] = None
    published_agent_ids: Optional[frozenset[int]] = None

    def is_all(self) -> bool:
        return False

    def is_none(self) -> bool:
        return True

    def is_by_categories(self) -> bool:
        return False

    def includes_mcp(self) -> bool:
        return False

    def includes_custom_api(self) -> bool:
        return False

    def includes_published_agent(self) -> bool:
        return False

    def compute_allowed_names(self, all_tools: List[Any]) -> Optional[frozenset[str]]:
        # Empty frozenset signals "filter to []" -- factory drops
        # every tool returned by the registry. Distinct from
        # ``None`` (ALL mode, keep everything).
        return frozenset()


@dataclass(frozen=True)
class _SpecByCategories(ToolSelectionSpec):
    """BY_CATEGORIES mode — filtered build.

    ``categories`` is normally non-empty. The one valid empty-category
    state is workforce manager injection: no ordinary categories, but
    explicit ``name_extras`` worker-agent tools.
    """

    categories: frozenset[str] = field(default_factory=frozenset)
    mcp_servers: Optional[frozenset[str]] = None
    custom_api_ids: Optional[frozenset[int]] = None
    published_agent_ids: Optional[frozenset[int]] = None
    # Workforce worker tool name injection. Only meaningful in
    # BY_CATEGORIES mode (in ALL the full set already includes
    # them; in NONE everything is rejected).
    name_extras: frozenset[str] = field(default_factory=frozenset)
    # The pre-derivation user input, used by ``compute_allowed_names``
    # to tell apart "user picked plain 'mcp' (admit ALL mcp tools)"
    # from "user picked only 'mcp:<server>' (from_raw added 'mcp' to
    # categories for the registry skip, but the name filter should
    # NOT broaden to every mcp tool)". Set by :meth:`from_raw`. For
    # direct construction the default is the same as ``categories``
    # (so direct-construction callers without sub-category derivation
    # behave as the legacy single-dataclass spec did).
    _user_picked: Optional[frozenset[str]] = None

    def __post_init__(self) -> None:
        if not self.categories and not self.name_extras:
            raise ValueError(
                "_SpecByCategories requires non-empty categories or "
                "name_extras. "
                "Use ToolSelectionSpec.from_raw() with empty / None "
                "categories to get _SpecAll, or pass "
                "explicit_none=True for _SpecNone."
            )
        # Defense against direct-construction bypass: if categories
        # carry the agent-builder ``mcp:<server>`` sub-category form,
        # the parallel ``mcp_servers`` field must be populated for the
        # MCP creator's per-server filter to work. ``from_raw`` derives
        # both consistently; a caller that direct-constructs with
        # ``mcp:<server>`` in categories but ``mcp_servers=None`` would
        # land in an inconsistent state.
        mcp_sub_categories = {
            c for c in self.categories if isinstance(c, str) and c.startswith("mcp:")
        }
        if mcp_sub_categories and self.mcp_servers is None:
            raise ValueError(
                f"_SpecByCategories with mcp:<server> sub-categories "
                f"({sorted(mcp_sub_categories)}) requires mcp_servers "
                f"to be set (the parallel per-server filter). "
                f"Construct via ToolSelectionSpec.from_raw(tool_categories=...) "
                f"to derive both fields consistently."
            )

    def is_all(self) -> bool:
        return False

    def is_none(self) -> bool:
        return False

    def is_by_categories(self) -> bool:
        return True

    def _user_categories(self) -> frozenset[str]:
        """Return the pre-derivation user-picked category set.

        ``from_raw`` records the user's original list here so the
        name-filter step can tell "user said plain 'mcp'" (admit all
        mcp tools) apart from "user said only 'mcp:<server>'" (admit
        only that server's tools). For direct construction without
        a ``_user_picked`` arg, fall back to ``categories`` — that
        matches the legacy single-dataclass behaviour where
        ``categories`` was the user input verbatim.
        """
        return self._user_picked if self._user_picked is not None else self.categories

    def includes_mcp(self) -> bool:
        # Need "mcp" in categories; otherwise the registry-level
        # skip would have caught it but creators with no static
        # categories annotation can still consult this method.
        if "mcp" not in self.categories:
            return False
        if self.mcp_servers is not None and len(self.mcp_servers) == 0:
            return False
        return True

    def includes_custom_api(self) -> bool:
        # P2 fix: Custom API tools live under the "other" category.
        # If the caller restricts categories without including
        # "other", custom API tools cannot survive the post-build
        # name filter, so the creator's DB lookup is wasted I/O.
        if "other" not in self.categories:
            return False
        if self.custom_api_ids is not None and len(self.custom_api_ids) == 0:
            return False
        return True

    def includes_published_agent(self) -> bool:
        if self.name_extras:
            return True
        if "agent" not in self.categories:
            return False
        if self.published_agent_ids is not None and len(self.published_agent_ids) == 0:
            return False
        return True

    def compute_allowed_names(self, all_tools: List[Any]) -> Optional[frozenset[str]]:
        """Filter ``all_tools`` by ``categories`` + add ``name_extras``.

        Folds the matching logic of the retired
        ``select_allowed_tool_names_from_categories`` helper plus
        the ``_merge_workforce_tool_names`` workforce-extra step
        into a single dispatch.

        Duck-typed access to ``tool.metadata.category`` keeps this
        module free of any Tool / AbstractBaseTool import.
        """
        names: Set[str] = set()
        for tool in all_tools:
            if not (hasattr(tool, "metadata") and hasattr(tool.metadata, "category")):
                continue
            tool_name = getattr(tool, "name", None)
            if not isinstance(tool_name, str):
                continue
            category = str(tool.metadata.category.value)

            # Plain category match. Note ``from_raw`` may add "mcp" /
            # "other" to ``categories`` when the user picked a
            # ``mcp:<server>`` sub-category — that is for the
            # registry-level skip, not a name-level admit. The
            # ``categories`` frozenset distinguishes the two cases by
            # carrying the original raw strings:
            #
            #   from_raw(["mcp"])           -> {"mcp"}
            #     → user explicitly asked for ALL mcp tools; admit
            #   from_raw(["mcp:Gmail"])     -> {"mcp", "other", "mcp:Gmail"}
            #     → user asked for one server; do NOT broaden to all mcp
            #   from_raw(["mcp", "mcp:X"])  -> {"mcp", "other", "mcp:X"}
            #     → "mcp" plain entry wins → admit all mcp tools
            #
            # So a tool whose category is "mcp" / "other" admits when
            # the *plain* string is in ``categories``; a tool whose
            # category is "mcp:<server>" wouldn't exist (servers don't
            # have their own category) -- this branch only runs once
            # per tool. ``_raw_user_categories`` is set by ``from_raw``
            # to the pre-derivation user input; for direct construction
            # it falls back to ``categories``.
            user_picked = self._user_categories()
            if category in ("mcp", "other"):
                if category in user_picked:
                    names.add(tool_name)
                    continue
                # No plain "mcp" / "other" picked; fall through to
                # mcp:<server> sub-category matching below.
            elif category in self.categories:
                names.add(tool_name)
                continue

            # "mcp:<server>" sub-category — match mcp_<server>_*
            if category == "mcp":
                for cat_spec in self.categories:
                    if not cat_spec.startswith("mcp:"):
                        continue
                    server = (
                        cat_spec.split(":", 1)[1].replace(" ", "_").replace("-", "_")
                    )
                    if tool_name.lower().startswith(f"mcp_{server.lower()}_"):
                        names.add(tool_name)
                        break

            # "mcp:<server>" sub-category — match api_<server>_call
            # (Custom-API tools surface under the "other" category but
            # the user expresses them through the same mcp:<server> tag.)
            elif category == "other":
                for cat_spec in self.categories:
                    if not cat_spec.startswith("mcp:"):
                        continue
                    server = (
                        cat_spec.split(":", 1)[1].replace(" ", "_").replace("-", "_")
                    )
                    if tool_name.lower() == f"api_{server.lower()}_call":
                        names.add(tool_name)
                        break

        # Inject workforce worker tool names (only meaningful in
        # this mode; ``from_raw`` zeroes out for ALL / NONE).
        return frozenset(names | self.name_extras)
