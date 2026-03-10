from queue import Queue
from typing import Any, Tuple

task_queue = Queue()

def _create_task_item(uid: str, data: dict) -> Tuple[str, dict]:
    """
    Creates a task item from UID and data.

    Args:
        uid: A unique string that serves as an identifier for each task.
        data: A dictionary containing the details and parameters associated
              with the task.

    Returns:
        A tuple representing the task item.
    """
    return (uid, data)

def _add_item_to_queue(item: Any, queue_obj: Queue) -> bool:
    """
    Adds an item to the specified queue.

    Args:
        item: The item to be added to the queue.
        queue_obj: The queue to which the item will be added.

    Returns:
        True if the item was successfully added, False otherwise.
    """
    try:
        queue_obj.put(item)
        return True
    except Exception:
        # Optionally, log the exception here
        return False

def push_task(uid: str, data: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.

    This function is responsible for integrating new tasks into a global queue
    for future execution.

    Args:
        uid: A unique string that serves as an identifier for each task.
        data: A dictionary containing the details and parameters associated
              with the task.

    Returns:
        Indicates whether the task was successfully added to the queue (True)
        or not (False).
    """
    task_item = _create_task_item(uid, data)
    return _add_item_to_queue(task_item, task_queue)
