from queue import Queue
from typing import Tuple, Any

task_queue = Queue()


def _prepare_task_item(identifier: str, details: dict) -> Tuple[str, dict]:
    """
    Prepares the task item tuple from identifier and details.
    Pure function.

    Args:
        identifier: The task's unique identifier.
        details: The task's details.

    Returns:
        A tuple containing the identifier and details.
    """
    return (identifier, details)


def _add_item_to_queue(task_item: Tuple[Any, ...], queue_obj: Queue) -> bool:
    """
    Adds a prepared task item to the specified queue.
    Pure function with respect to its inputs, though it mutates queue_obj.

    Args:
        task_item: The item to add to the queue.
        queue_obj: The queue to add the item to.

    Returns:
        True if the item was added successfully, False otherwise.
        (Note: queue.put() doesn't typically fail unless queue is full and blocking=False,
         or other specific Queue implementations. For standard Queue, it blocks or raises Full.
         Here, we assume success unless an unexpected error occurs, caught by the caller.)
    """
    queue_obj.put(task_item)
    return True


def enqueue_task(identifier: str, details: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.

    This function is responsible for integrating new tasks into a global queue
    for future execution.

    Args:
        identifier: A unique string that serves as an identifier for each task.
        details: A dictionary containing the details and parameters associated 
                 with the task.

    Returns:
        Indicates whether the task was successfully added to the queue (True) 
        or not (False).
    """
    try:
        task_item = _prepare_task_item(identifier, details)
        return _add_item_to_queue(task_item, task_queue)
    except Exception:
        # Log the exception here if logging is set up
        return False
