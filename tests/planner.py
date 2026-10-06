"""
Tests for ``invoke.planner`` (explicit, non-executing invocation plans).

Run with::

    python -m pytest tests/planner.py -v

On this Windows worktree the repo-wide conftest needs a ``termios`` shim
and a broken globally-installed pytest plugin must be bypassed, so the
acceptance run used::

    PYTHONPATH=D:\\Codex\\pytest-shims PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        python -m pytest -p relaxed tests/planner.py -v
"""

from unittest.mock import Mock

import pytest

from invoke import (
    AmbiguousPlanError,
    Collection,
    Executor,
    MissingParameterError,
    Plan,
    PlanError,
    Planner,
    Task,
    call,
    task,
    when,
)
from invoke.parser import ParserContext
from invoke.planner import PENDING, READY


def _named(name, **kwargs):
    # A Task wrapping a Mock body, with an explicit name for stable output.
    return Task(Mock(name=name), name=name, **kwargs)


def _three_task_collection():
    """
    The three-task example from invoke.planner's docstring, with recording
    bodies so tests can prove nothing executed during planning.
    """
    ran = []

    @task
    def setup(c, clean=False):
        ran.append(("setup", clean))

    @task(pre=[call(setup, clean=True)])
    def build(c):
        ran.append(("build",))

    @task(pre=[call(setup)], post=[when(setup, lambda: False)])
    def release(c, target="pypi"):
        ran.append(("release", target))

    ns = Collection(setup, build, release)
    return ns, ran, (setup, build, release)


class Planner_:
    class goal_parsing:
        def string_names(self):
            t = _named("t")
            coll = Collection(t)
            plan = Planner(coll).plan("t")
            assert [x.task for x in plan.calls] == [t]
            assert plan.calls[0].called_as == "t"

        def name_kwargs_tuples(self):
            t = _named("t")
            coll = Collection(t)
            plan = Planner(coll).plan(("t", {"x": 1}))
            assert plan.calls[0].kwargs == {"x": 1}

        def parser_contexts(self):
            t = _named("t")
            coll = Collection(t)
            plan = Planner(coll).plan(ParserContext(name="t"))
            assert [x.task for x in plan.calls] == [t]

        def call_and_task_objects(self):
            t = _named("t")
            coll = Collection(t)
            plan = Planner(coll).plan(call(t, x=2), t)
            # Same task, different args -> two distinct identities
            assert len(plan.calls) == 2
            assert plan.calls[0].kwargs == {"x": 2}

        def default_task_when_no_goals_given(self):
            t = _named("t")
            coll = Collection()
            coll.add_task(t, default=True)
            plan = Planner(coll).plan()
            assert [x.task for x in plan.calls] == [t]

        def unknown_names_are_ambiguous(self):
            with pytest.raises(AmbiguousPlanError):
                Planner(Collection()).plan("nope")

        def uninterpretable_goals_are_ambiguous(self):
            with pytest.raises(AmbiguousPlanError):
                Planner(Collection()).plan(12345)

    class pre_post_parameter_binding:
        def pre_tasks_expand_before_goal(self):
            pre = _named("pre")
            main = _named("main", pre=[pre])
            plan = Planner(Collection(pre, main)).plan("main")
            assert [x.name for x in plan.calls] == ["pre", "main"]

        def post_tasks_expand_after_goal(self):
            post = _named("post")
            main = _named("main", post=[post])
            plan = Planner(Collection(post, main)).plan("main")
            assert [x.name for x in plan.calls] == ["main", "post"]

        def string_entries_resolve_against_collection(self):
            pre = _named("pre")
            main = _named("main", pre=["pre"])
            plan = Planner(Collection(pre, main)).plan("main")
            assert [x.name for x in plan.calls] == ["pre", "main"]

        def unknown_string_entries_are_ambiguous(self):
            main = _named("main", pre=["ghost"])
            with pytest.raises(AmbiguousPlanError):
                Planner(Collection(main)).plan("main")

        def call_objects_bind_their_arguments(self):
            pre = _named("pre")
            main = _named("main", pre=[call(pre, clean=True)])
            plan = Planner(Collection(pre, main)).plan("main")
            assert plan.calls[0].kwargs == {"clean": True}

        def nested_pre_chains_expand_depth_first(self):
            leaf = _named("leaf")
            mid = _named("mid", pre=[leaf])
            top = _named("top", pre=[mid])
            plan = Planner(Collection(leaf, mid, top)).plan("top")
            assert [x.name for x in plan.calls] == ["leaf", "mid", "top"]

        def cyclic_chains_raise_plan_error(self):
            a = _named("a")
            b = _named("b", pre=[a])
            a.pre = [b]
            with pytest.raises(PlanError):
                Planner(Collection(a, b)).plan("a")

    class validation_before_any_execution:
        def _needs_collection(self):
            def needs(c, target):
                pass

            needed = Task(needs, name="needs")
            ok = _named("ok")
            coll = Collection()
            coll.add_task(needed)
            coll.add_task(ok)
            return coll, needed, ok

        def missing_required_params_raise(self):
            coll, needed, ok = self._needs_collection()
            with pytest.raises(MissingParameterError) as info:
                Planner(coll).plan("needs")
            assert info.value.task_name == "needs"
            assert info.value.missing == ["target"]

        def missing_params_mean_zero_task_executions(self):
            coll, needed, ok = self._needs_collection()
            with pytest.raises(MissingParameterError):
                # Even the valid goal must not execute: validation of the
                # whole plan happens before any task body may run.
                Planner(coll).plan("ok", "needs")
            assert needed.times_called == 0
            assert ok.times_called == 0
            assert not ok.body.called

        def supplied_params_satisfy_requirements(self):
            coll, needed, ok = self._needs_collection()
            plan = Planner(coll).plan(("needs", {"target": "x"}))
            assert len(plan.calls) == 1

        def unknown_kwargs_are_ambiguous(self):
            def real(c, known=1):
                pass

            t = Task(real, name="real")
            coll = Collection(t)
            with pytest.raises(AmbiguousPlanError):
                Planner(coll).plan(("real", {"bogus": 1}))

        def doubly_bound_arguments_are_ambiguous(self):
            def real(c, x):
                pass

            t = Task(real, name="real")
            coll = Collection(t)
            with pytest.raises(AmbiguousPlanError):
                Planner(coll).plan(call(t, "posval", x="kwval"))

        def too_many_positionals_are_ambiguous(self):
            def real(c, x=1):
                pass

            t = Task(real, name="real")
            coll = Collection(t)
            with pytest.raises(AmbiguousPlanError):
                Planner(coll).plan(call(t, 1, 2, 3))

    class dedup_identity_includes_arguments:
        def _two_goal_collection(self, first_kwargs, second_kwargs):
            setup = _named("setup")
            one = _named("one", pre=[call(setup, **first_kwargs)])
            two = _named("two", pre=[call(setup, **second_kwargs)])
            return Collection(setup, one, two)

        def shared_pre_with_different_args_yields_two_identities(self):
            coll = self._two_goal_collection(
                {"clean": True}, {"clean": False}
            )
            plan = Planner(coll).plan("one", "two")
            assert [x.name for x in plan.calls] == [
                "setup",
                "one",
                "setup",
                "two",
            ]
            setups = [x for x in plan.calls if x.name == "setup"]
            assert len(setups) == 2
            assert setups[0].identity != setups[1].identity
            assert len(set(plan.identities)) == 4

        def shared_pre_with_same_args_dedupes(self):
            coll = self._two_goal_collection(
                {"clean": True}, {"clean": True}
            )
            plan = Planner(coll).plan("one", "two")
            assert [x.name for x in plan.calls] == ["setup", "one", "two"]
            assert len(set(plan.identities)) == 3

        def same_goal_same_args_dedupes(self):
            t = _named("t")
            coll = Collection(t)
            plan = Planner(coll).plan(("t", {"x": 1}), ("t", {"x": 1}))
            assert len(plan.calls) == 1

        def same_goal_different_args_does_not_dedupe(self):
            t = _named("t")
            coll = Collection(t)
            plan = Planner(coll).plan(("t", {"x": 1}), ("t", {"x": 2}))
            assert len(plan.calls) == 2

        def positional_args_are_part_of_identity(self):
            def real(c, x):
                pass

            t = Task(real, name="t")
            coll = Collection(t)
            plan = Planner(coll).plan(call(t, "a"), call(t, "b"))
            assert len(plan.calls) == 2

    class conditional_posts:
        def statically_true_conditions_enter_plan(self):
            post = _named("post")
            main = _named("main", post=[when(post, True)])
            plan = Planner(Collection(post, main)).plan("main")
            assert [x.name for x in plan.calls] == ["main", "post"]
            assert plan.pending == []

        def statically_false_conditions_never_enter_plan(self):
            post = _named("post")
            main = _named("main", post=[when(post, False)])
            plan = Planner(Collection(post, main)).plan("main")
            assert [x.name for x in plan.calls] == ["main"]
            assert plan.pending == []

        def dynamic_conditions_are_pending_not_executed(self):
            condition = Mock(return_value=True)
            post = _named("post")
            main = _named("main", post=[when(post, condition)])
            plan = Planner(Collection(post, main)).plan("main")
            # Not in the executable sequence...
            assert [x.name for x in plan.calls] == ["main"]
            # ...but marked pending (undecided)...
            assert len(plan.pending) == 1
            pending = plan.pending[0]
            assert pending.task is post
            assert pending.status == PENDING
            assert pending.reason
            # ...and neither the condition nor any task body ran to decide.
            condition.assert_not_called()
            assert post.times_called == 0
            assert main.times_called == 0

        def dynamic_conditions_may_wrap_string_names(self):
            post = _named("post")
            main = _named("main", post=[when("post", lambda: True)])
            plan = Planner(Collection(post, main)).plan("main")
            assert [x.name for x in plan.pending] == ["post"]

        def invalid_condition_values_are_ambiguous(self):
            post = _named("post")
            main = _named("main", post=[when(post, "maybe")])
            with pytest.raises(AmbiguousPlanError):
                Planner(Collection(post, main)).plan("main")

        def unconditional_posts_always_enter_plan(self):
            post = _named("post")
            main = _named("main", post=[post])
            plan = Planner(Collection(post, main)).plan("main")
            assert [x.name for x in plan.calls] == ["main", "post"]

    class provenance_dependency_chains:
        def sources_record_goal_pre_post_chain(self):
            pre = _named("pre")
            post = _named("post")
            main = _named("main", pre=[pre], post=[post])
            plan = Planner(Collection(pre, post, main)).plan("main")
            pre_call, main_call, post_call = plan.calls
            assert main_call.source == ("goal:main",)
            assert pre_call.source == ("goal:main", "pre:pre")
            assert post_call.source == ("goal:main", "post:post")

        def nested_chains_keep_full_path(self):
            leaf = _named("leaf")
            mid = _named("mid", pre=[leaf])
            top = _named("top", pre=[mid])
            plan = Planner(Collection(leaf, mid, top)).plan("top")
            assert plan.calls[0].source == (
                "goal:top",
                "pre:mid",
                "pre:leaf",
            )

        def deduped_calls_keep_all_contributing_chains(self):
            setup = _named("setup")
            one = _named("one", pre=[call(setup, clean=True)])
            two = _named("two", pre=[call(setup, clean=True)])
            plan = Planner(Collection(setup, one, two)).plan("one", "two")
            setups = [x for x in plan.calls if x.name == "setup"]
            assert len(setups) == 1
            assert setups[0].sources == [
                ("goal:one", "pre:setup"),
                ("goal:two", "pre:setup"),
            ]

        def pending_calls_keep_their_chain_too(self):
            post = _named("post")
            main = _named("main", post=[when(post, lambda: True)])
            plan = Planner(Collection(post, main)).plan("main")
            assert plan.pending[0].source == ("goal:main", "post:post")

    class non_execution_and_status_quo:
        def planning_never_executes_task_bodies(self):
            pre = _named("pre")
            main = _named("main", pre=[pre])
            Planner(Collection(pre, main)).plan("main")
            assert not pre.body.called
            assert not main.body.called
            assert pre.times_called == 0
            assert main.times_called == 0

        def ready_calls_are_marked_ready(self):
            main = _named("main")
            plan = Planner(Collection(main)).plan("main")
            assert [x.status for x in plan.calls] == [READY]

        def executor_behavior_is_unchanged(self):
            # Planning is strictly opt-in: Executor still expands, dedupes
            # and executes exactly as before.
            pre = _named("pre")
            main = _named("main", pre=[pre])
            coll = Collection(pre, main)
            Planner(coll).plan("main")
            Executor(collection=coll).execute("main")
            assert pre.body.called
            assert main.body.called

        def collection_plan_method_matches_planner(self):
            main = _named("main")
            coll = Collection(main)
            via_method = coll.plan("main")
            via_class = Planner(coll).plan("main")
            assert isinstance(via_method, Plan)
            assert via_method.calls == via_class.calls

    class three_task_example:
        def docstring_example_behaves_as_documented(self):
            ns, ran, (setup, build, release) = _three_task_collection()
            plan = ns.plan("build", ("release", {"target": "testpypi"}))
            # Shared 'setup' pre-task appears twice: arguments differ.
            assert [x.task.name for x in plan.calls] == [
                "setup",
                "build",
                "setup",
                "release",
            ]
            assert plan.calls[0].kwargs == {"clean": True}
            assert plan.calls[2].kwargs == {}
            # Dynamic conditional post is pending, not executed.
            assert [x.task.name for x in plan.pending] == ["setup"]
            # Nothing ran while planning.
            assert ran == []
            assert setup.times_called == 0
            assert build.times_called == 0
            assert release.times_called == 0

        def docstring_example_dedupes_identical_shared_pre(self):
            ran = []

            @task
            def setup(c, clean=False):
                ran.append("setup")

            @task(pre=[call(setup)])
            def build(c):
                ran.append("build")

            @task(pre=[call(setup)])
            def release(c):
                ran.append("release")

            ns = Collection(setup, build, release)
            plan = ns.plan("build", "release")
            assert [x.task.name for x in plan.calls] == [
                "setup",
                "build",
                "release",
            ]
            assert ran == []
