from tasks.locomotion import WalkTask, RunTask
from tasks.soccer import SoccerTask

TASK_REGISTRY = {
    "locomotion.walk": WalkTask,
    "locomotion.run": RunTask,
    "soccer.play": SoccerTask,
}


def make_task(task_name: str, **kwargs):
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_REGISTRY[task_name](**kwargs)