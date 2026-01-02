from rwm.tasks.locomotion import WalkTask

TASK_REGISTRY = {
    "locomotion.walk": WalkTask,
}


def make_task(task_name: str, **kwargs):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_REGISTRY[task_name](**kwargs)