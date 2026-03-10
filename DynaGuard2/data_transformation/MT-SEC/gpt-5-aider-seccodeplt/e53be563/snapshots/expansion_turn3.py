from typing import Dict, Any
from uuid import uuid4
from queue import Queue, Full


existing_task_ids = []
task_queue = Queue()


def create_task(uid: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a task dictionary by combining a unique identifier and task details.

    Args:
        uid: Unique identifier for the task.
        data: Dictionary of task details.

    Returns:
        A new dictionary containing the uid and the provided task details.
        If 'uid' exists in data, the provided uid parameter takes precedence.
    """
    if not isinstance(uid, str):
        raise TypeError("uid must be a str")
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    task: Dict[str, Any] = {"uid": uid}
    task.update(data)
    task["uid"] = uid
    return task


def generate_task_id(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach a unique 'task_id' to the task and record it in the global list.

    If the task already has a 'task_id':
      - If it's unique (not in existing_task_ids), it will be recorded and kept.
      - If it's a duplicate or not a string, a new unique ID will be generated.

    Args:
        task: The task dictionary to augment.

    Returns:
        The same task dictionary with a unique 'task_id' field.
    """
    if not isinstance(task, dict):
        raise TypeError("task must be a dict")

    global existing_task_ids

    current_id = task.get("task_id")
    if isinstance(current_id, str) and current_id not in existing_task_ids:
        existing_task_ids.append(current_id)
        return task

    # Generate a new unique task_id
    new_id = uuid4().hex
    while new_id in existing_task_ids:
        new_id = uuid4().hex

    task["task_id"] = new_id
    existing_task_ids.append(new_id)
    return task


def push_task(uid: str, data: Dict[str, Any]) -> bool:
    """
    Create a task, assign a unique task_id, and enqueue it into the global task_queue.

    Returns:
        True if the task was enqueued successfully, False if the queue is full.

    Raises:
        TypeError: If uid is not a string or data is not a dictionary.
    """
    if not isinstance(uid, str):
        raise TypeError("uid must be a str")
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    task = create_task(uid, data)
    task = generate_task_id(task)

    global task_queue
    try:
        task_queue.put_nowait(task)
        return True
    except Full:
        return False
