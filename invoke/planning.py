"""
Explicit, non-executing task-call planning.

This module implements explicit *call plans* for `.Collection`: given one or
more *goals* (task names, ``(name, kwargs)`` tuples, `.ParserContext` or
`.Call` objects) it statically resolves the full call graph - goals, pre
tasks and conditional post tasks - binds pre/post parameters, and validates
the result **before any task body executes**.

Nothing in this module ever executes a task body to obtain an answer.
Conditions whose outcome cannot be determined statically are represented as
*pending* plan entries instead of being resolved by running tasks.

The classic `.Executor` API is intentionally untouched: planning is an
opt-in, explicit operation via `Collection.plan <.Collection.plan>`.

.. versionadded:: 3.1
"""

from inspect import Parameter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from .parser import ParserContext
from .tasks import Call, Task

if TYPE_CHECKING:
    from .collection import Collection


class PlanError(Exception):
    """
    Base class for errors detected while building a `.TaskPlan`.

    .. versionadded:: 3.1
    """

    pass


class MissingParameters(PlanError):
    """
    A planned call is missing one or more required (default-less) parameters.

    .. versionadded:: 3.1
    """

    pass


class AmbiguousPlan(PlanError):
    """
    A planned call is ambiguous: unknown, duplicated or excess parameters.

    .. versionadded:: 3.1
    """

    pass


class Undetermined(Exception):
    """
    Raise from a condition callable when its outcome is not statically known.

    The planner marks the guarded call *pending* instead of executing
    anything (including tasks) to obtain the answer.

    .. versionadded:: 3.1
    """

    pass


class ConditionalCall(Call):
    """
    A `.Call` guarded by a planning-time ``condition``.

    ``condition`` must be a callable accepting the *parent* call's bound
    keyword arguments (a dict) and returning a boolean:

    * ``True``: the guarded call enters the plan;
    * ``False``: it is omitted from the plan;
    * raising `.Undetermined` (or using a `.Task` as the condition, which the
      planner refuses to execute): the guarded call enters the plan marked
      *pending*.

    Under the classic `.Executor` this object behaves exactly like a plain
    `.Call` - the condition is ignored and the call executes unconditionally
    - so existing behavior is unchanged unless one explicitly plans.

    .. versionadded:: 3.1
    """

    def __init__(
        self,
        task: "Task",
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        called_as: Optional[str] = None,
        args: Optional[Tuple[str, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(task, called_as=called_as, args=args, kwargs=kwargs)
        self.condition = condition


def when(
    task: "Task",
    condition: Callable[[Dict[str, Any]], bool],
    *args: Any,
    **kwargs: Any,
) -> ConditionalCall:
    """
    Declare a conditional (typically post-) task call, for use with planning.

    Analogous to `.call`, but the resulting `.ConditionalCall` only enters a
    `.TaskPlan` when ``condition`` (called with the parent call's bound
    keyword arguments) is satisfied. Example::

        @task(post=[when(notify, lambda kw: kw.get("release"))])
        def build(c, release=False):
            ...

    .. versionadded:: 3.1
    """
    return ConditionalCall(task, condition=condition, args=args, kwargs=kwargs)


def _freeze(value: Any) -> Any:
    """
    Turn ``value`` into a hashable form suitable for dedup identities.
    """
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze(v) for v in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


class PlannedCall:
    """
    A single statically-planned task call, with its dependency chain.

    Attributes of note:

    * ``kind``: one of ``"goal"``, ``"pre"`` or ``"post"``;
    * ``parent``: the `.PlannedCall` whose task declared this one as a
      pre/post task (``None`` for goals), preserving the dependency chain;
    * ``bound``: the normalized parameter mapping (positional arguments
      folded into their parameter names) actually used for dedup identity;
    * ``pending`` / ``reason``: set when the call's inclusion could not be
      determined statically (e.g. an `.Undetermined` condition).

    .. versionadded:: 3.1
    """

    def __init__(
        self,
        task: "Task",
        called_as: Optional[str],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        bound: Dict[str, Any],
        kind: str,
        parent: Optional["PlannedCall"],
        pending: bool = False,
        reason: Optional[str] = None,
    ) -> None:
        self.task = task
        self.called_as = called_as
        self.args = args
        self.kwargs = kwargs
        self.bound = bound
        self.kind = kind
        self.parent = parent
        self.pending = pending
        self.reason = reason

    @property
    def identity(self) -> Tuple[str, Any]:
        """
        Dedup identity: the task name plus the actual bound parameters.

        Two planned calls to the same task are only duplicates when their
        concrete parameters match; the task name alone is never sufficient.

        .. versionadded:: 3.1
        """
        return (self.task.name, _freeze(self.bound))

    @property
    def chain(self) -> Tuple["PlannedCall", ...]:
        """
        The dependency chain from the root goal down to this call.

        .. versionadded:: 3.1
        """
        links: List["PlannedCall"] = []
        node: Optional["PlannedCall"] = self
        while node is not None:
            links.append(node)
            node = node.parent
        return tuple(reversed(links))

    def __repr__(self) -> str:
        flags = ""
        if self.pending:
            flags = ", pending: {}".format(self.reason or "yes")
        return "<PlannedCall {!r} kind={} bound={!r}{}>".format(
            self.task.name, self.kind, self.bound, flags
        )


class TaskPlan:
    """
    The result of planning: an ordered, deduplicated list of `.PlannedCall`.

    Planning never executes task bodies. If any `.PlanError` was detected
    (missing or ambiguous parameters, unknown tasks, ...) ``ok`` is false and
    `counts` reports zero for every known task, reflecting that nothing may
    run until the errors are fixed.

    .. versionadded:: 3.1
    """

    def __init__(
        self,
        calls: List[PlannedCall],
        errors: List[PlanError],
        task_names: List[str],
    ) -> None:
        self.calls = list(calls)
        self.errors = list(errors)
        self.task_names = list(task_names)

    @property
    def ok(self) -> bool:
        """
        Whether the plan is free of planning errors.
        """
        return not self.errors

    @property
    def ready(self) -> List[PlannedCall]:
        """
        Planned calls whose execution is statically determined.
        """
        return [c for c in self.calls if not c.pending]

    @property
    def pending(self) -> List[PlannedCall]:
        """
        Planned calls whose inclusion could not be statically determined.
        """
        return [c for c in self.calls if c.pending]

    def counts(self) -> Dict[str, int]:
        """
        Map every known task name to its number of planned executions.

        Pending calls do not count (they may never run). If the plan has any
        errors, **all** task counts are zero: nothing is considered runnable
        until the plan validates cleanly.
        """
        tallies = {name: 0 for name in self.task_names}
        if self.errors:
            return tallies
        for pcall in self.ready:
            tallies[pcall.task.name] = tallies.get(pcall.task.name, 0) + 1
        return tallies

    def names(self) -> List[str]:
        """
        Task names of all planned calls, in execution order.
        """
        return [c.task.name for c in self.calls]

    def __iter__(self) -> Iterator[PlannedCall]:
        return iter(self.calls)

    def __len__(self) -> int:
        return len(self.calls)

    def __repr__(self) -> str:
        return "<TaskPlan ok={} calls={!r} errors={!r}>".format(
            self.ok, self.names(), self.errors
        )


def _bind(
    task: "Task", args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], List[str], List[str], bool]:
    """
    Bind ``args``/``kwargs`` against ``task``'s signature, without executing.

    Returns ``(bound, missing, unknown, duplicates, excess_positionals)``.
    """
    sig = task.argspec(task.body)
    params = list(sig.parameters.values())
    K = Parameter  # shorthand for inspect.Parameter kind constants
    positional_slots = [
        p
        for p in params
        if p.kind in (K.POSITIONAL_ONLY, K.POSITIONAL_OR_KEYWORD)
    ]
    has_var_positional = any(p.kind is K.VAR_POSITIONAL for p in params)
    has_var_keyword = any(p.kind is K.VAR_KEYWORD for p in params)
    keyword_names = {
        p.name
        for p in params
        if p.kind in (K.POSITIONAL_OR_KEYWORD, K.KEYWORD_ONLY)
    }
    bound: Dict[str, Any] = {}
    for param, value in zip(positional_slots, args):
        bound[param.name] = value
    excess = len(args) > len(positional_slots) and not has_var_positional
    unknown: List[str] = []
    duplicates: List[str] = []
    for name, value in kwargs.items():
        if name in bound:
            duplicates.append(name)
        elif name in keyword_names or has_var_keyword:
            bound[name] = value
        else:
            unknown.append(name)
    required_kinds = (
        K.POSITIONAL_ONLY,
        K.POSITIONAL_OR_KEYWORD,
        K.KEYWORD_ONLY,
    )
    missing = [
        p.name
        for p in params
        if p.default is K.empty
        and p.kind in required_kinds
        and p.name not in bound
    ]
    return bound, missing, unknown, duplicates, excess


def _desc(call: Call) -> str:
    return "'{}'".format(call.called_as or call.task.name)


class Planner:
    """
    Builds explicit `.TaskPlan` objects for a `.Collection`, without ever
    executing task bodies.

    .. versionadded:: 3.1
    """

    def __init__(self, collection: "Collection") -> None:
        self.collection = collection

    def plan(
        self,
        *goals: Union[str, Tuple[str, Dict[str, Any]], ParserContext, Call],
    ) -> TaskPlan:
        """
        Build a `.TaskPlan` for ``goals``.

        Goal formats mirror `.Executor.execute`: task name strings (dotted
        namespace syntax included), ``(name, kwargs)`` tuples,
        `.ParserContext` objects, or `.Call` objects. With no goals at all,
        the collection's default task is planned, if any.

        :returns: a `.TaskPlan`; inspect ``.ok``/``.errors`` before trusting
            ``.counts``.
        """
        errors: List[PlanError] = []
        task_names: List[str] = []
        planned: List[PlannedCall] = []
        for call in self._normalize(goals, errors, task_names):
            self._expand(
                call,
                kind="goal",
                parent=None,
                planned=planned,
                errors=errors,
                task_names=task_names,
            )
        # Deduplicate by identity (task name + actual bound parameters),
        # preserving first-seen order.
        deduped: List[PlannedCall] = []
        seen = set()
        for pcall in planned:
            if pcall.identity in seen:
                continue
            seen.add(pcall.identity)
            deduped.append(pcall)
        return TaskPlan(calls=deduped, errors=errors, task_names=task_names)

    def _normalize(
        self,
        goals: Tuple[Any, ...],
        errors: List[PlanError],
        task_names: List[str],
    ) -> List[Call]:
        calls: List[Call] = []
        if not goals:
            default = self.collection.default
            if default is None:
                return calls
            goals = (default,)
        for goal in goals:
            if isinstance(goal, Call):
                call = goal
                if call.called_as is None:
                    call = call.clone(with_={"called_as": call.task.name})
            else:
                name: Optional[str]
                if isinstance(goal, str):
                    name, kwargs = goal, {}
                elif isinstance(goal, ParserContext):
                    name, kwargs = goal.name, goal.as_kwargs
                else:
                    name, kwargs = goal
                try:
                    task = self.collection[name]
                except Exception as e:
                    errors.append(
                        PlanError("Unknown task {!r}: {}".format(name, e))
                    )
                    continue
                call = Call(task, kwargs=kwargs, called_as=name)
            self._register(task_names, call.task.name)
            calls.append(call)
        return calls

    def _register(self, task_names: List[str], name: str) -> None:
        if name not in task_names:
            task_names.append(name)

    def _expand(
        self,
        call: Call,
        kind: str,
        parent: Optional[PlannedCall],
        planned: List[PlannedCall],
        errors: List[PlanError],
        task_names: List[str],
    ) -> Optional[PlannedCall]:
        # Validate parameters before anything is added to the plan; no task
        # body ever runs here.
        bound, missing, unknown, duplicates, excess = _bind(
            call.task, call.args, call.kwargs
        )
        ok = True
        if missing:
            errors.append(
                MissingParameters(
                    "Call to {} is missing required parameter(s): {}".format(
                        _desc(call), ", ".join(missing)
                    )
                )
            )
            ok = False
        ambiguity = []
        if unknown:
            ambiguity.append(
                "unknown parameter(s): {}".format(", ".join(sorted(unknown)))
            )
        if duplicates:
            ambiguity.append(
                "parameter(s) given both positionally and by keyword: "
                "{}".format(", ".join(duplicates))
            )
        if excess:
            ambiguity.append("too many positional arguments")
        if ambiguity:
            errors.append(
                AmbiguousPlan(
                    "Call to {} is ambiguous: {}".format(
                        _desc(call), "; ".join(ambiguity)
                    )
                )
            )
            ok = False
        if not ok:
            return None
        me = PlannedCall(
            task=call.task,
            called_as=call.called_as,
            args=call.args,
            kwargs=call.kwargs,
            bound=bound,
            kind=kind,
            parent=parent,
        )
        # Pre tasks (recursively), then self, then post tasks - mirroring the
        # executor's expansion order so the plan preserves the real
        # dependency chain.
        for entry in call.task.pre:
            self._expand_entry(entry, "pre", me, planned, errors, task_names)
        planned.append(me)
        for entry in call.task.post:
            self._expand_entry(entry, "post", me, planned, errors, task_names)
        return me

    def _expand_entry(
        self,
        entry: Any,
        kind: str,
        parent: PlannedCall,
        planned: List[PlannedCall],
        errors: List[PlanError],
        task_names: List[str],
    ) -> None:
        if isinstance(entry, Call):
            call = entry
        elif isinstance(entry, Task):
            call = Call(entry)
        else:
            errors.append(
                PlanError(
                    "Unsupported {} entry {!r} on task {!r}; expected a Task "
                    "or Call".format(kind, entry, parent.task.name)
                )
            )
            return
        self._register(task_names, call.task.name)
        if isinstance(entry, ConditionalCall):
            verdict = self._evaluate(entry.condition, parent)
            if verdict is False:
                # Declared condition not satisfied: never enters the plan.
                return
            if verdict is None:
                # Dynamically undeterminable: mark pending, do not execute
                # anything (certainly not a task) to obtain the answer.
                bound, _, _, _, _ = _bind(call.task, call.args, call.kwargs)
                planned.append(
                    PlannedCall(
                        task=call.task,
                        called_as=call.called_as,
                        args=call.args,
                        kwargs=call.kwargs,
                        bound=bound,
                        kind=kind,
                        parent=parent,
                        pending=True,
                        reason="condition undetermined at planning time",
                    )
                )
                return
        self._expand(call, kind, parent, planned, errors, task_names)

    def _evaluate(
        self,
        condition: Optional[Callable[[Dict[str, Any]], bool]],
        parent: PlannedCall,
    ) -> Optional[bool]:
        if condition is None:
            return True
        if isinstance(condition, Task):
            # Never execute a task to obtain an answer at planning time.
            return None
        try:
            return bool(condition(dict(parent.bound)))
        except Undetermined:
            return None
