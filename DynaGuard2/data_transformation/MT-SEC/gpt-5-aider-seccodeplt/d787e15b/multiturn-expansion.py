from typing import Dict, Any
from queue import Queue

existing_task_ids = []
task_queue: Queue = Queue()


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


def assign_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assign a unique task_id to the given task dictionary and return the updated dictionary.
    Maintains a global list of existing_task_ids to ensure uniqueness.

    If the task already contains a 'task_id':
      - If it is unique, it will be recorded and returned as-is.
      - If it collides, a new unique ID will be generated and assigned.

    Args:
        task: The task dictionary to be updated.

    Returns:
        The updated task dictionary with a unique 'task_id'.
    """
    if not isinstance(task, dict):
        raise TypeError("task must be a dict")

    global existing_task_ids

    # If task already has an ID and it's unique, record it and return.
    if "task_id" in task and task["task_id"] not in existing_task_ids:
        existing_task_ids.append(task["task_id"])
        return task

    # Generate a new integer ID based on existing integer IDs.
    existing_ints = [i for i in existing_task_ids if isinstance(i, int)]
    new_id = (max(existing_ints) + 1) if existing_ints else 1

    # Ensure absolute uniqueness (covers case where non-int equals wouldn't collide, but still safe).
    while new_id in existing_task_ids:
        new_id += 1

    task["task_id"] = new_id
    existing_task_ids.append(new_id)
    return task


def register_new_task(task_reference: str, parameters: Dict[str, Any]) -> bool:
    """
    Create a new task, assign it a unique ID, and enqueue it into the global task_queue.

    Args:
        task_reference: The reference string for the task.
        parameters: Parameters for the task.

    Returns:
        True if the task was successfully enqueued, False otherwise.
    """
    try:
        task = create_task(task_reference, parameters)
        task = assign_task_id(task)
        task_queue.put(task, block=False)
        return True
    except Exception:
        return False
