from typing import Dict, Any

def create_task(task_reference: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a task object by combining task reference and parameters.

    Args:
        task_reference: The reference string for the task.
        parameters: A dictionary of parameters for the task.

    Returns:
        A dictionary representing the task object.
    """
    return {
        "task_reference": task_reference,
        "parameters": parameters,
    }
