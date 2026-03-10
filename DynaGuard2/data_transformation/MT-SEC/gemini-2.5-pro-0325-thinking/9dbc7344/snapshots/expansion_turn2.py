from typing import Dict, Any, List

task_ids: List[str] = []

def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique task_id to the task and records it.

    Args:
        task: The task dictionary.

    Returns:
        The task dictionary updated with a 'task_id'.
    """
    new_id = f"task_{len(task_ids) + 1}"
    task_ids.append(new_id)
    task['task_id'] = new_id
    return task

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
