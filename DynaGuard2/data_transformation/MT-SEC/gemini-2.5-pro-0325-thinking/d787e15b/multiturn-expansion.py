from typing import Dict, Any, List
import uuid
from queue import Queue

existing_task_ids: List[str] = []
task_queue: Queue = Queue()

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

def assign_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique task ID to the task and records it.

    Args:
        task: The task dictionary.

    Returns:
        The task dictionary updated with a 'task_id'.
    """
    while True:
        task_id = str(uuid.uuid4())
        if task_id not in existing_task_ids:
            break
    task["task_id"] = task_id
    existing_task_ids.append(task_id)
    return task

def register_new_task(task_reference: str, parameters: Dict[str, Any]) -> bool:
    """
    Creates a new task, assigns it an ID, and adds it to the global task queue.

    Args:
        task_reference: The reference string for the task.
        parameters: A dictionary of parameters for the task.

    Returns:
        True if the task was successfully added to the queue, False otherwise.
    """
    try:
        new_task = create_task(task_reference, parameters)
        task_with_id = assign_task_id(new_task)
        task_queue.put(task_with_id)
        return True
    except Exception:
        # In a real-world scenario, specific exceptions should be caught and logged.
        return False
