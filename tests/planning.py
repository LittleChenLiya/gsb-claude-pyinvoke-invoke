"""
Tests for explicit task-call planning (invoke.planning / Collection.plan).

Run with: python -m pytest tests/planning.py -v
"""

from invoke import (
    AmbiguousPlan,
    Call,
    Collection,
    Executor,
    MissingParameters,
    PlanError,
    Undetermined,
    call,
    task,
    when,
)
from invoke.planning import ConditionalCall, Planner, TaskPlan


def _make_collection(ran):
    """
    Build a three-task collection (setup/build/notify) plus extras.

    ``ran`` records every task body execution, proving planning runs none.
    """

    @task
    def setup(c, clean=False):
        ran.append(("setup", clean))

    @task
    def notify(c, channel="email"):
        ran.append(("notify", channel))

    @task(
        pre=[call(setup, clean=True)],
        post=[when(notify, lambda kw: kw.get("release"))],
    )
    def build(c, release=False):
        ran.append(("build", release))

    @task(pre=[call(setup, clean=False)])
    def package(c):
        ran.append("package")

    @task(pre=[call(setup, clean=True)])
    def rebuild(c):
        ran.append("rebuild")

    @task
    def deploy(c, target):
        ran.append(("deploy", target))

    return Collection(setup, notify, build, package, rebuild, deploy)


class TestGoalsAndExpansion:
    def setup_method(self):
        self.ran = []
        self.ns = _make_collection(self.ran)

    def test_plans_goal_with_pre_and_satisfied_post(self):
        plan = self.ns.plan(("build", {"release": True}))
        assert plan.ok
        assert plan.names() == ["setup", "build", "notify"]
        assert plan.counts() == {"setup": 1, "build": 1, "notify": 1}
        # Tasks not referenced by any goal simply count as zero.
        assert plan.counts().get("package", 0) == 0

    def test_post_only_enters_plan_when_declared_condition_satisfied(self):
        plan = self.ns.plan("build")  # release left at its default
        assert plan.ok
        assert "notify" not in plan.names()
        assert plan.counts()["notify"] == 0
        assert plan.names() == ["setup", "build"]

    def test_multiple_goals_plan_in_order(self):
        plan = self.ns.plan("package", "rebuild")
        assert plan.names() == ["setup", "package", "setup", "rebuild"]

    def test_unknown_goal_is_a_plan_error(self):
        plan = self.ns.plan("nope")
        assert not plan.ok
        assert isinstance(plan.errors[0], PlanError)
        assert all(count == 0 for count in plan.counts().values())

    def test_no_goals_plans_default_task(self):
        @task(default=True)
        def main(c):
            pass

        ns = Collection(main)
        plan = ns.plan()
        assert plan.ok
        assert plan.names() == ["main"]


class TestDedupIdentityIncludesActualParameters:
    def setup_method(self):
        self.ran = []
        self.ns = _make_collection(self.ran)

    def test_same_pre_with_different_params_yields_two_identities(self):
        plan = self.ns.plan("build", "package")
        setups = [c for c in plan if c.task.name == "setup"]
        assert len(setups) == 2
        assert setups[0].identity != setups[1].identity
        assert plan.counts()["setup"] == 2

    def test_same_pre_with_same_params_dedupes_to_one(self):
        plan = self.ns.plan("build", "rebuild")
        setups = [c for c in plan if c.task.name == "setup"]
        assert len(setups) == 1
        assert plan.counts()["setup"] == 1
        # ...but the distinct goals themselves both remain.
        assert plan.counts()["build"] == 1
        assert plan.counts()["rebuild"] == 1

    def test_identity_is_not_task_name_alone(self):
        plan = self.ns.plan("build", "package")
        names = [c.task.name for c in plan]
        # Two setup entries share a name but not an identity.
        assert names.count("setup") == 2


class TestValidationBeforeAnyExecution:
    def setup_method(self):
        self.ran = []
        self.ns = _make_collection(self.ran)

    def test_missing_params_make_all_task_counts_zero(self):
        plan = self.ns.plan("deploy")  # requires 'target'
        assert not plan.ok
        assert isinstance(plan.errors[0], MissingParameters)
        counts = plan.counts()
        assert counts["deploy"] == 0
        assert all(count == 0 for count in counts.values())

    def test_missing_params_in_pre_still_zeroes_everything(self):
        @task
        def needs_arg(c, value):
            pass

        @task(pre=[needs_arg])
        def top(c):
            pass

        ns = Collection(needs_arg, top)
        plan = ns.plan("top")
        assert not plan.ok
        assert isinstance(plan.errors[0], MissingParameters)
        assert plan.counts() == {"needs_arg": 0, "top": 0}

    def test_unknown_kwarg_is_ambiguous(self):
        plan = self.ns.plan(("deploy", {"target": "x", "bogus": 1}))
        assert not plan.ok
        assert isinstance(plan.errors[0], AmbiguousPlan)
        assert all(count == 0 for count in plan.counts().values())

    def test_duplicate_positional_and_keyword_is_ambiguous(self):
        plan = self.ns.plan(call(self.ns["deploy"], "x", target="y"))
        assert not plan.ok
        assert isinstance(plan.errors[0], AmbiguousPlan)

    def test_planning_never_executes_task_bodies(self):
        self.ns.plan("build", "package", "rebuild")
        self.ns.plan("deploy")
        assert self.ran == []
        for t in self.ns.tasks.values():
            assert t.times_called == 0


class TestUndeterminedConditionsArePending:
    def test_undetermined_condition_marks_post_pending(self):
        def condition(kw):
            raise Undetermined("needs runtime information")

        @task
        def ping(c):
            pass

        @task(post=[when(ping, condition)])
        def pong(c):
            pass

        ns = Collection(ping, pong)
        plan = ns.plan("pong")
        assert plan.ok  # pending is not an error
        assert [c.task.name for c in plan.pending] == ["ping"]
        assert plan.counts()["ping"] == 0  # pending does not count
        assert plan.counts()["pong"] == 1

    def test_task_as_condition_is_not_executed_for_an_answer(self):
        ran = []

        @task
        def decide(c):
            ran.append("decide")
            return True

        @task
        def ping(c):
            ran.append("ping")

        @task(post=[when(ping, decide)])
        def pong(c):
            ran.append("pong")

        ns = Collection(decide, ping, pong)
        plan = ns.plan("pong")
        assert [c.task.name for c in plan.pending] == ["ping"]
        assert ran == []  # no task executed to resolve the condition
        assert decide.times_called == 0


class TestDependencyChainIsPreserved:
    def setup_method(self):
        self.ran = []
        self.ns = _make_collection(self.ran)

    def test_pre_and_post_link_back_to_declaring_goal(self):
        plan = self.ns.plan(("build", {"release": True}))
        setup, build, notify = plan.calls
        assert build.parent is None
        assert setup.parent is build
        assert notify.parent is build
        assert setup.kind == "pre"
        assert notify.kind == "post"
        assert build.kind == "goal"

    def test_chain_walks_from_root_goal(self):
        plan = self.ns.plan("build")
        setup = plan.calls[0]
        assert [c.task.name for c in setup.chain] == ["build", "setup"]

    def test_nested_pre_chains_span_multiple_levels(self):
        @task
        def leaf(c):
            pass

        @task(pre=[leaf])
        def mid(c):
            pass

        @task(pre=[mid])
        def top(c):
            pass

        ns = Collection(leaf, mid, top)
        plan = ns.plan("top")
        leaf_call = plan.calls[0]
        assert [c.task.name for c in leaf_call.chain] == ["top", "mid", "leaf"]


class TestOriginalInvokeBehaviorIsUnchanged:
    def setup_method(self):
        self.ran = []
        self.ns = _make_collection(self.ran)

    def test_executor_still_executes_normally(self):
        Executor(collection=self.ns).execute("build")
        # Classic executor: pre runs, goal runs, and the conditional post is
        # treated as a plain (unconditional) call.
        assert self.ran == [
            ("setup", True),
            ("build", False),
            ("notify", "email"),
        ]

    def test_executor_ignores_conditions_on_conditional_calls(self):
        ran = []

        @task
        def ping(c):
            ran.append("ping")

        @task(post=[when(ping, lambda kw: False)])
        def pong(c):
            ran.append("pong")

        Executor(collection=Collection(ping, pong)).execute("pong")
        assert ran == ["pong", "ping"]

    def test_executor_dedupes_by_call_equality_as_before(self):
        Executor(collection=self.ns).execute("build", "rebuild")
        # Classic executor dedupes identical pre calls (same args).
        assert self.ran.count(("setup", True)) == 1

    def test_collection_without_plan_call_is_untouched(self):
        # Merely building a collection changes nothing.
        assert isinstance(self.ns, Collection)
        assert "build" in self.ns.task_names


class TestPlannerObject:
    def setup_method(self):
        self.ran = []
        self.ns = _make_collection(self.ran)

    def test_planner_wraps_collection(self):
        planner = Planner(self.ns)
        assert planner.collection is self.ns

    def test_plan_returns_task_plan(self):
        assert isinstance(self.ns.plan("build"), TaskPlan)

    def test_conditional_call_is_a_call(self):
        c = ConditionalCall(self.ns["notify"], condition=lambda kw: True)
        assert isinstance(c, Call)
