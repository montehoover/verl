from typing import Dict, Any

def create_task(uid: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a task dictionary by combining a unique identifier and task data.

    Args:
        uid: The unique identifier for the task.
        data: A dictionary containing the details of the task.

    Returns:
        A dictionary representing the task, including its uid and data.
    """
    task = {"uid": uid}
    task.update(data)
    return task
