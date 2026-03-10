from typing import Dict, Any

def create_task(task_identifier: str, task_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a task object by combining a task identifier and its details.

    Args:
        task_identifier: The unique identifier for the task.
        task_details: A dictionary containing the details of the task.

    Returns:
        A dictionary representing the task object, including its identifier and details.
    """
    task_object: Dict[str, Any] = {
        'id': task_identifier,
        **task_details
    }
    return task_object
