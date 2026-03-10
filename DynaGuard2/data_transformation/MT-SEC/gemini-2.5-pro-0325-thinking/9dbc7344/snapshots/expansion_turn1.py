from typing import Dict, Any

def create_task(task_identifier: str, task_content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Encapsulates task details.

    Args:
        task_identifier: The unique identifier for the task.
        task_content: A dictionary containing the specifics of the task.

    Returns:
        A dictionary combining the task identifier and its content.
    """
    return {
        "task_id": task_identifier,
        "content": task_content
    }
