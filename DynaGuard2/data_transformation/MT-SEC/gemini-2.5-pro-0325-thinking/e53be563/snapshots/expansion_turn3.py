from typing import Dict, Any, List
from queue import Queue

existing_task_ids: List[int] = []
task_queue: Queue = Queue()

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

def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds a unique task_id to the task dictionary and records it.

    Args:
        task: The task dictionary.

    Returns:
        The updated task dictionary with a 'task_id'.
    """
    global existing_task_ids
    task_id = len(existing_task_ids)
    task['task_id'] = task_id
    existing_task_ids.append(task_id)
    return task

def push_task(uid: str, data: Dict[str, Any]) -> bool:
    """
    Creates a task, assigns it an ID, and adds it to the global task queue.

    Args:
        uid: The unique identifier for the task.
        data: A dictionary containing the details of the task.

    Returns:
        True if the task was successfully added to the queue, False otherwise.
    """
    try:
        task = create_task(uid, data)
        task_with_id = generate_task_id(task)
        task_queue.put(task_with_id)
        return True
    except Exception:
        # Optionally, log the exception here
        return False
