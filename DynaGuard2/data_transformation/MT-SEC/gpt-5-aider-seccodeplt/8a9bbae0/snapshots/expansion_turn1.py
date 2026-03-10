from typing import Dict, Any

def create_task(task_identifier: str, task_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a task object by combining the task identifier and task details.

    Args:
        task_identifier: Unique identifier for the task.
        task_details: Arbitrary details describing the task.

    Returns:
        A dictionary representing the task, with the identifier and a copy of the details.
    """
    if not isinstance(task_identifier, str):
        raise TypeError("task_identifier must be a str")
    if not isinstance(task_details, dict):
        raise TypeError("task_details must be a dict")

    # Use a shallow copy to prevent external mutations from affecting the task object.
    details_copy: Dict[str, Any] = dict(task_details)

    return {
        "task_identifier": task_identifier,
        "task_details": details_copy,
    }
