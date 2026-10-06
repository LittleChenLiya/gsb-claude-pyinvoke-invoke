# Examples

## `three_task_plan.py` — explicit task-call planning

A three-task example (`setup` / `build` / `notify`) of the explicit,
non-executing planning API (`Collection.plan`, `invoke.planning`):

- goals are parsed into a `TaskPlan` before any task body executes;
- pre/post parameters are bound and validated up front (missing or
  ambiguous parameters become plan errors and zero out all task counts);
- shared pre tasks dedupe by identity = task name **plus actual
  parameters**, so `call(setup, clean=True)` and `call(setup, clean=False)`
  are two distinct planned calls while identical ones collapse;
- conditional post tasks (`when(...)`) only enter the plan when their
  declared condition is satisfied; conditions that cannot be determined
  statically are marked *pending* instead of being resolved by executing
  tasks;
- each planned call keeps its dependency chain (`PlannedCall.parent` /
  `.chain`);
- the classic `Executor` API is unchanged — planning is strictly opt-in.

Run the example:

```sh
python examples/three_task_plan.py
```

Run the feature's tests:

```sh
python -m pytest tests/planning.py -v
```
