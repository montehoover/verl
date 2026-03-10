from typing import Dict, Any

def create_task(identifier: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a task representation by combining an identifier and details.

    Args:
        identifier: The unique identifier for the task.
        details: A dictionary containing the task's details.

    Returns:
        A dictionary representing the task, including its identifier and details.
    """
    task = {
        "identifier": identifier,
        "details": details
    }
    return task
