from typing import Dict, Any

def create_task(identifier: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a task representation combining an identifier and details.

    Args:
        identifier: The unique string identifier for the task.
        details: A dictionary of task details.

    Returns:
        A dictionary representing the task.
    """
    return {
        "identifier": identifier,
        "details": dict(details),
    }
