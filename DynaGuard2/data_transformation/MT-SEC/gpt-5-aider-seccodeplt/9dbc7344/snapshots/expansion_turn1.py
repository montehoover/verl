from typing import Dict, Any


def create_task(task_identifier: str, task_content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a task dictionary that encapsulates the task identifier and its content.

    Args:
        task_identifier: A unique string identifying the task.
        task_content: A dictionary containing the task's specifics.

    Returns:
        A dictionary combining the task identifier and content.
    """
    return {
        "task_identifier": task_identifier,
        "task_content": dict(task_content),  # shallow copy to avoid external mutation
    }
