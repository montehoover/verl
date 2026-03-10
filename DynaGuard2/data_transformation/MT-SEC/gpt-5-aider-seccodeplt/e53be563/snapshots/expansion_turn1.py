from typing import Dict, Any


def create_task(uid: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a task dictionary by combining a unique identifier and task details.

    Args:
        uid: Unique identifier for the task.
        data: Dictionary of task details.

    Returns:
        A new dictionary containing the uid and the provided task details.
        If 'uid' exists in data, the provided uid parameter takes precedence.
    """
    if not isinstance(uid, str):
        raise TypeError("uid must be a str")
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    task: Dict[str, Any] = {"uid": uid}
    task.update(data)
    task["uid"] = uid
    return task
