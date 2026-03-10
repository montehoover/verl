from typing import Dict, Any
import uuid
from queue import Queue, Full

task_id_list: list[str] = []
task_queue = Queue()

def create_task(task_identifier: str, task_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a task object by combining the task identifier and task details.

    Args:
        task_identifier: Unique identifier for the task.
        task_details: Arbitrary details describing the task.

    Returns:
        A dictionary representing the task, with the identifier and a copy of the details.
    """
    if not isinstance(task_identifier, str):
        raise TypeError("task_identifier must be a str")
    if not isinstance(task_details, dict):
        raise TypeError("task_details must be a dict")

    # Use a shallow copy to prevent external mutations from affecting the task object.
    details_copy: Dict[str, Any] = dict(task_details)

    return {
        "task_identifier": task_identifier,
        "task_details": details_copy,
    }

def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assign a unique task_id to the task and track it in a global list.

    If the task already has a 'task_id', it will be ensured to be tracked.

    Args:
        task: The task dictionary to update.

    Returns:
        The task dictionary with a 'task_id'.
    """
    if not isinstance(task, dict):
        raise TypeError("task must be a dict")

    existing_id = task.get("task_id")
    if isinstance(existing_id, str):
        if existing_id not in task_id_list:
            task_id_list.append(existing_id)
        return task

    # Generate a unique ID and ensure no collision with existing IDs.
    new_id = uuid.uuid4().hex
    while new_id in task_id_list:
        new_id = uuid.uuid4().hex

    updated_task: Dict[str, Any] = dict(task)
    updated_task["task_id"] = new_id
    task_id_list.append(new_id)

    return updated_task

def enqueue_task(task_identifier: str, task_details: Dict[str, Any]) -> bool:
    """
    Enqueue a task into the global task_queue.

    Args:
        task_identifier: The identifier for the task.
        task_details: The details associated with the task.

    Returns:
        True if the task was successfully enqueued, False otherwise.
    """
    try:
        task = create_task(task_identifier, task_details)
        task_with_id = generate_task_id(task)
        task_queue.put_nowait(task_with_id)
        return True
    except (TypeError, Full):
        return False
