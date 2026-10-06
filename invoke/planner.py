"""
Explicit, non-executing task invocation plans.

This module implements an opt-in "planning" layer on top of `.Collection`
(see `Collection.plan <.Collection.plan>`). It parses goals, binds pre- and
post-task parameters, and validates the resulting calls -- all *before*, and
without, executing any task body:

* Goals may be task-name strings, ``(name, kwargs)`` tuples,
  `.ParserContext` objects, `.Call` objects or raw `.Task` objects
  (mirroring the inputs accepted by `.Executor.execute`).
* Pre/post task entries may be `.Task` objects, `.Call` objects (see
  `.tasks.call`), task-name strings, or `Conditional` wrappers (see `when`).
* Missing required parameters and ambiguous bindings (unknown task names,
  unknown or doubly-bound arguments, cyclic pre/post chains) raise
  `MissingParameterError` / `AmbiguousPlanError` before anything executes.
* Deduplication identity includes the *actual arguments* of each call, not
  just the task name: two goals referencing the same pre-task with different
  arguments yield two distinct planned calls, while identical calls collapse
  into one.
* Conditional post-tasks (see `when`) only enter the plan when their
  declared condition is statically satisfied. Conditions that cannot be
  determined without running code (e.g. callables) are marked *pending*
  instead of being executed to obtain an answer.
* Every planned call records its provenance -- the goal/pre/post chain that
  brought it into the plan -- in `PlannedCall.source` (and, after
  deduplication, all contributing chains in `PlannedCall.sources`).

Nothing in this module changes the behavior of `.Executor` or the ``invoke``
CLI; planning is strictly additive and opt-in.

Three-task example
==================

Given a three-task collection where two goals share a pre-task, bound with
different arguments, plus a dynamically conditional post-task::

    from invoke import Collection, call, task, when

    @task
    def setup(c, clean=False):
        print("setup, clean={}".format(clean))

    @task(pre=[call(setup, clean=True)])
    def build(c):
        print("build")

    @task(pre=[call(setup)], post=[when(setup, lambda: False)])
    def release(c, target="pypi"):
        print("release to {}".format(target))

    ns = Collection(setup, build, release)
    plan = ns.plan("build", ("release", {"target": "testpypi"}))

    [x.task.name for x in plan.calls]
    # -> ['setup', 'build', 'setup', 'release']: the shared 'setup' pre-task
    # appears twice because its *arguments* differ between the two goals.
    [x.task.name for x in plan.pending]
    # -> ['setup']: the conditional post-task cannot be statically resolved,
    # so it is marked pending; its condition is never executed to decide.

Run the tests for this module with::

    python -m pytest tests/planner.py -v

.. versionadded:: 3.1
"""

import inspect
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    TYPE_CHECKING,
    Tuple,
)

from .parser import ParserContext
from .tasks import Call, Task

if TYPE_CHECKING:
    from .collection import Collection


#: Status of a planned call that is fully determined and ready to execute.
READY = "ready"
#: Status of a planned call whose inclusion cannot be statically determined.
PENDING = "pending"


class PlanError(Exception):
    """
    Base class for errors raised while building an explicit plan.

    .. versionadded:: 3.1
    """


class MissingParameterError(PlanError):
    """
    A planned call is missing one or more required parameters.

    Attributes: ``task_name`` (the offending task's name) and ``missing`` (a
    list of the unbound required parameter names).

    .. versionadded:: 3.1
    """

    def __init__(self, task_name: str, missing: List[str]) -> None:
        self.task_name = task_name
        self.missing = list(missing)
        super().__init__(
            "Task {!r} is missing required parameter(s): {}".format(
                task_name, ", ".join(repr(x) for x in self.missing)
            )
        )


class AmbiguousPlanError(PlanError):
    """
    A goal or call could not be resolved unambiguously at plan time.

    Raised for unknown task names, unknown or doubly-bound arguments,
    uninterpretable goal/pre/post entries, and invalid condition values.

    .. versionadded:: 3.1
    """


def _freeze(value: Any) -> Any:
    """
    Turn ``value`` into a hashable structure suitable for dedup identities.
    """
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


class Conditional:
    """
    A pre/post task entry guarded by a declared condition. See `when`.

    .. versionadded:: 3.1
    """

    def __init__(self, obj: Any, condition: Any) -> None:
        self.obj = obj
        self.condition = condition

    def __repr__(self) -> str:
        return "<Conditional {!r} if {!r}>".format(self.obj, self.condition)


def when(obj: Any, condition: Any) -> Conditional:
    """
    Declare that pre/post entry ``obj`` only joins a plan if ``condition``.

    ``obj`` is anything normally allowed in a ``pre``/``post`` list (a
    `.Task`, a `.Call`, or a task-name string). ``condition`` is interpreted
    statically at plan time:

    * ``True``/``False``: the entry is unconditionally included/excluded.
    * A callable (or any other dynamically-computed value): the entry cannot
      be resolved without executing code, so it is marked
      `pending <PlannedCall.status>` in the resulting `.Plan`. The condition
      is **never called** -- planning must not execute code to get answers.

    .. versionadded:: 3.1
    """
    return Conditional(obj, condition)


class PlannedCall:
    """
    A single, fully-bound call within an explicit `.Plan`.

    Analogous to `.Call`, but plan-oriented: it carries a ``source``
    provenance chain and a ``status`` instead of knowing how to execute.

    .. versionadded:: 3.1
    """

    def __init__(
        self,
        task: "Task",
        called_as: Optional[str] = None,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        source: Tuple[str, ...] = (),
        status: str = READY,
        reason: Optional[str] = None,
    ) -> None:
        self.task = task
        self.called_as = called_as
        self.args = tuple(args or ())
        self.kwargs = dict(kwargs or {})
        #: Provenance chain, e.g. ``("goal:build", "pre:setup")``.
        self.source = tuple(source)
        #: All provenance chains (deduplication merges duplicates' chains).
        self.sources: List[Tuple[str, ...]] = []
        if self.source:
            self.sources.append(self.source)
        #: `READY` or `PENDING`.
        self.status = status
        #: Human-readable explanation when ``status`` is `PENDING`.
        self.reason = reason

    @property
    def name(self) -> str:
        """
        The name this call was referred to by (falling back to task name).
        """
        return self.called_as or self.task.name

    @property
    def identity(self) -> Tuple["Task", Any, Any]:
        """
        Deduplication identity: the task *plus its actual arguments*.

        Two calls to the same task with different ``args``/``kwargs`` have
        different identities; identical calls share one.
        """
        return (self.task, _freeze(self.args), _freeze(self.kwargs))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlannedCall):
            return NotImplemented
        return self.identity == other.identity

    def __hash__(self) -> int:
        return hash(self.identity)

    def __repr__(self) -> str:
        return "<{} {!r} args={!r} kwargs={!r} status={!r}>".format(
            self.__class__.__name__,
            self.name,
            self.args,
            self.kwargs,
            self.status,
        )


class Plan:
    """
    An explicit, ordered, non-executed invocation plan.

    Attributes:

    * ``calls``: list of `PlannedCall` in execution order -- deduplicated by
      `identity <PlannedCall.identity>` and fully validated. This is what
      would execute, in order, if the plan were carried out.
    * ``pending``: list of `PlannedCall` whose inclusion could not be
      statically determined (e.g. dynamically conditional post-tasks). These
      are *not* part of the executable sequence.

    .. versionadded:: 3.1
    """

    def __init__(
        self,
        calls: Optional[List[PlannedCall]] = None,
        pending: Optional[List[PlannedCall]] = None,
    ) -> None:
        self.calls = list(calls or [])
        self.pending = list(pending or [])

    @property
    def tasks(self) -> List["Task"]:
        """
        The `.Task` objects of all ready calls, in execution order.
        """
        return [x.task for x in self.calls]

    @property
    def identities(self) -> List[Tuple["Task", Any, Any]]:
        """
        The deduplication identities of all ready calls, in order.
        """
        return [x.identity for x in self.calls]

    def __iter__(self) -> Iterator[PlannedCall]:
        return iter(self.calls)

    def __len__(self) -> int:
        return len(self.calls)

    def __contains__(self, call: object) -> bool:
        return call in self.calls

    def __repr__(self) -> str:
        return "<Plan calls={} pending={}>".format(
            [x.name for x in self.calls], [x.name for x in self.pending]
        )


class Planner:
    """
    Builds explicit `.Plan` objects from a `.Collection` and a set of goals.

    .. versionadded:: 3.1
    """

    def __init__(self, collection: "Collection") -> None:
        self.collection = collection

    def plan(self, *goals: Any) -> Plan:
        """
        Parse ``goals`` into a validated, deduplicated `.Plan`.

        No task body (and no dynamic condition) is executed by this method;
        invalid plans raise `PlanError` subclasses before anything runs.

        :param goals:
            Task-name strings, ``(name, kwargs)`` tuples, `.ParserContext`
            objects, `.Call` objects or `.Task` objects. When empty, the
            collection's default task is planned, if any.

        :returns: A `.Plan`.

        .. versionadded:: 3.1
        """
        calls = self._normalize(goals)
        ready: List[PlannedCall] = []
        pending: List[PlannedCall] = []
        for call in calls:
            self._expand(call, ready, pending, chain=call.source, stack=())
        # Validate everything that would execute, *before* anything could.
        for call in ready:
            self._validate(call)
        return Plan(calls=self._dedupe(ready), pending=self._dedupe(pending))

    def _normalize(self, goals: Tuple[Any, ...]) -> List[PlannedCall]:
        calls = []
        for goal in goals:
            calls.append(self._goal_to_call(goal))
        if not goals and self.collection.default is not None:
            name = self.collection.default
            task = self.collection[name]
            calls = [PlannedCall(task, source=("goal:{}".format(name),))]
        return calls

    def _goal_to_call(self, goal: Any) -> PlannedCall:
        if isinstance(goal, PlannedCall):
            return goal
        if isinstance(goal, str):
            return PlannedCall(
                self._lookup(goal),
                called_as=goal,
                source=("goal:{}".format(goal),),
            )
        if isinstance(goal, ParserContext):
            if goal.name is None:
                raise AmbiguousPlanError(
                    "Cannot plan unnamed parser context {!r}".format(goal)
                )
            return PlannedCall(
                self._lookup(goal.name),
                called_as=goal.name,
                kwargs=goal.as_kwargs,
                source=("goal:{}".format(goal.name),),
            )
        if isinstance(goal, Call):
            return PlannedCall(
                goal.task,
                called_as=goal.called_as,
                args=goal.args,
                kwargs=goal.kwargs,
                source=("goal:{}".format(goal.task.name),),
            )
        if isinstance(goal, Task):
            return PlannedCall(goal, source=("goal:{}".format(goal.name),))
        if (
            isinstance(goal, (tuple, list))
            and len(goal) == 2
            and isinstance(goal[0], str)
        ):
            name, kwargs = goal
            return PlannedCall(
                self._lookup(name),
                called_as=name,
                kwargs=kwargs,
                source=("goal:{}".format(name),),
            )
        raise AmbiguousPlanError(
            "Cannot interpret goal {!r} as a task call".format(goal)
        )

    def _lookup(self, name: str) -> "Task":
        try:
            return self.collection[name]
        except (KeyError, ValueError) as e:
            raise AmbiguousPlanError(
                "No task named {!r} in collection".format(name)
            ) from e

    def _entry_to_call(
        self, entry: Any, relation: str, chain: Tuple[str, ...]
    ) -> PlannedCall:
        if isinstance(entry, PlannedCall):
            call = entry
        elif isinstance(entry, Call):
            call = PlannedCall(
                entry.task,
                called_as=entry.called_as,
                args=entry.args,
                kwargs=entry.kwargs,
            )
        elif isinstance(entry, Task):
            call = PlannedCall(entry)
        elif isinstance(entry, str):
            call = PlannedCall(self._lookup(entry), called_as=entry)
        else:
            raise AmbiguousPlanError(
                "Cannot interpret {} entry {!r} as a task call".format(
                    relation, entry
                )
            )
        call.source = chain + ("{}:{}".format(relation, call.name),)
        call.sources = [call.source]
        return call

    def _expand(
        self,
        call: PlannedCall,
        ready: List[PlannedCall],
        pending: List[PlannedCall],
        chain: Tuple[str, ...],
        stack: Tuple["Task", ...],
    ) -> None:
        if call.task in stack:
            raise PlanError(
                "Cyclic pre/post task chain involving {!r}".format(
                    call.task.name
                )
            )
        stack = stack + (call.task,)
        for entry in call.task.pre:
            self._expand_entry(entry, "pre", ready, pending, chain, stack)
        ready.append(call)
        for entry in call.task.post:
            self._expand_entry(entry, "post", ready, pending, chain, stack)

    def _expand_entry(
        self,
        entry: Any,
        relation: str,
        ready: List[PlannedCall],
        pending: List[PlannedCall],
        chain: Tuple[str, ...],
        stack: Tuple["Task", ...],
    ) -> None:
        condition: Any = True
        if isinstance(entry, Conditional):
            condition = entry.condition
            entry = entry.obj
        # Statically unsatisfied condition: never enters the plan at all.
        if isinstance(condition, bool) and not condition:
            return
        call = self._entry_to_call(entry, relation, chain)
        if isinstance(condition, bool):
            # Statically satisfied (or no condition declared): recurse.
            self._expand(call, ready, pending, call.source, stack)
        elif callable(condition):
            # Dynamic: mark pending; never execute the condition (or any
            # task) to obtain an answer, and do not expand its own deps.
            call.status = PENDING
            call.reason = (
                "dynamic condition {!r} cannot be resolved without "
                "executing code".format(condition)
            )
            pending.append(call)
        else:
            raise AmbiguousPlanError(
                "Cannot statically interpret condition {!r} for {} entry "
                "{!r}".format(condition, relation, call.name)
            )

    def _validate(self, call: PlannedCall) -> None:
        sig = call.task.argspec(call.task.body)
        params = list(sig.parameters.values())
        names = {p.name for p in params}
        kinds_pos = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        positional = [p for p in params if p.kind in kinds_pos]
        has_var_pos = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
        )
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params
        )
        bound = set()
        if len(call.args) > len(positional) and not has_var_pos:
            raise AmbiguousPlanError(
                "Task {!r} takes at most {} positional argument(s) but "
                "{} were given".format(
                    call.name, len(positional), len(call.args)
                )
            )
        for param in positional[: len(call.args)]:
            bound.add(param.name)
        for kwarg in call.kwargs:
            if kwarg in bound:
                raise AmbiguousPlanError(
                    "Task {!r} got multiple values for argument "
                    "{!r}".format(call.name, kwarg)
                )
            if kwarg not in names and not has_var_kw:
                raise AmbiguousPlanError(
                    "Task {!r} got an unexpected argument {!r}".format(
                        call.name, kwarg
                    )
                )
            bound.add(kwarg)
        missing = [
            p.name
            for p in params
            if p.default is inspect.Parameter.empty
            and p.kind in kinds_pos + (inspect.Parameter.KEYWORD_ONLY,)
            and p.name not in bound
        ]
        if missing:
            raise MissingParameterError(call.name, missing)

    def _dedupe(self, calls: List[PlannedCall]) -> List[PlannedCall]:
        deduped: List[PlannedCall] = []
        for call in calls:
            if call in deduped:
                existing = deduped[deduped.index(call)]
                for source in call.sources:
                    if source not in existing.sources:
                        existing.sources.append(source)
            else:
                deduped.append(call)
        return deduped
