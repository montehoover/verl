from typing import Dict, Any


def create_task(task_reference: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a task object by combining a task reference and parameters.

    Args:
        task_reference: A unique string to identify the task.
        parameters: A dictionary of parameters for the task.

    Returns:
        A dictionary representing the task.
    """
    if not isinstance(task_reference, str):
        raise TypeError("task_reference must be a str")
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dict")

    return {
        "task_reference": task_reference,
        "parameters": dict(parameters),
    }
