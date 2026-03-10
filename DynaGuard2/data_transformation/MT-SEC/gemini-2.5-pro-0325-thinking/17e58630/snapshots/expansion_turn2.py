from typing import Dict, Any, List

task_ids: List[int] = []

def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique task_id to the task and tracks it.

    Args:
        task: The task dictionary.

    Returns:
        The task dictionary updated with a 'task_id'.
    """
    global task_ids
    new_id = len(task_ids) + 1
    task['task_id'] = new_id
    task_ids.append(new_id)
    return task

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
