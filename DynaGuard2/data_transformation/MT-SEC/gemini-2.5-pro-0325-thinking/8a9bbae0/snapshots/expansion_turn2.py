from typing import Dict, Any, List

task_id_list: List[int] = []

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

def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique task_id to a task and tracks it.

    Args:
        task: The task dictionary.

    Returns:
        The task dictionary updated with a 'task_id'.
    """
    global task_id_list
    new_id = len(task_id_list)
    task['task_id'] = new_id
    task_id_list.append(new_id)
    return task
