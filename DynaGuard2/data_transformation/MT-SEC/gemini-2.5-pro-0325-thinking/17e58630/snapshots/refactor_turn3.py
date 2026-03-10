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

    This function is responsible for the direct operation of placing a task item
    into a given queue. It primarily involves a side effect on the queue object.

    Args:
        task_item: The item to add to the queue. This is typically a tuple
                   containing task information.
        queue_obj: The `queue.Queue` instance to which the item will be added.

    Returns:
        True if `queue_obj.put(task_item)` was called. Note: `queue.put()`
        itself does not return a value and will block if the queue is full
        (for bounded queues) unless it's a non-blocking put or a timeout
        is specified. This function assumes `put` is successful if no
        exception is raised. Any exceptions during the `put` operation
        should be handled by the caller.
    """
    queue_obj.put(task_item)
    return True


def enqueue_task(identifier: str, details: dict) -> bool:
    """
    Manages task addition to a shared queue in a concurrent processing environment.

    This function orchestrates the process of creating a task item from the
    provided identifier and details, and then adding this item to the global
    `task_queue`. It acts as the primary interface for enqueuing tasks.

    Args:
        identifier: A unique string that serves as an identifier for each
                    task.
        details: A dictionary containing the details and parameters
                 associated with the task.

    Returns:
        True if the task was successfully prepared and added to the queue,
        False if an exception occurred during the process.
    """
    try:
        task_item = _prepare_task_item(identifier, details)
        return _add_item_to_queue(task_item, task_queue)
    except Exception:
        # Log the exception here if logging is set up
        return False
