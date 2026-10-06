"""
Three-task example of explicit task-call planning (``Collection.plan``).

Tasks: ``setup`` (a shared pre task), ``build`` (the goal, with a
conditional ``notify`` post task) and ``notify``. Planning resolves the
goal/pre/post graph, binds parameters and validates everything *before* -
in fact without ever - executing a task body.

Run it directly to see a plan printed::

    python examples/three_task_plan.py

Run the feature's test suite with::

    python -m pytest tests/planning.py -v
"""

from invoke import Collection, call, task, when


@task
def setup(c, clean=False):
    """Shared pre task; its dedup identity includes ``clean``."""
    print("setup(clean={})".format(clean))


@task
def notify(c, channel="email"):
    """Post task; only planned when the declared condition holds."""
    print("notify(channel={})".format(channel))


@task(
    pre=[call(setup, clean=True)],
    # The post task only enters the plan when the parent's bound
    # 'release' parameter is truthy.
    post=[when(notify, lambda bound: bound.get("release"))],
)
def build(c, release=False):
    """Goal task declaring a parameterized pre and a conditional post."""
    print("build(release={})".format(release))


ns = Collection(setup, notify, build)


if __name__ == "__main__":
    for goals in [("build",), (("build", {"release": True}),)]:
        plan = ns.plan(*goals)
        print("goals: {!r}".format(goals))
        print("  ok: {}".format(plan.ok))
        print("  order: {}".format(plan.names()))
        print("  counts: {}".format(plan.counts()))
        for planned in plan:
            chain = " -> ".join(c.task.name for c in planned.chain)
            print("  - {} (kind={}, chain={})".format(
                planned.task.name, planned.kind, chain
            ))
