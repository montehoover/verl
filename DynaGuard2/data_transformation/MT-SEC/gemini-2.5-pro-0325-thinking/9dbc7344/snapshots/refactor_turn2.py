from queue import Queue
from typing import Tuple, Any

task_queue = Queue()

def _create_task_item(task_identifier: str, task_content: dict) -> Tuple[str, dict]:
    """
    Creates a task item tuple from the identifier and content.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.

    Returns:
        A tuple containing the task_identifier and task_content.
    """
    return (task_identifier, task_content)

def _enqueue_task_item(queue: Queue, task_item: Tuple[str, dict]) -> bool:
    """
    Enqueues a given task item into the provided queue.

    Args:
        queue: The queue to which the task item will be added.
        task_item: The task item (tuple) to be enqueued.

    Returns:
        True if the task is successfully enqueued, False otherwise.
    """
    try:
        queue.put(task_item)
        return True
    except Exception:
        # Catching a broad exception here as queue.Full might be relevant
        # for bounded queues, or other unforeseen issues during enqueuing.
        return False

def insert_task_to_queue(task_identifier: str, task_content: dict) -> bool:
    """
    Creates a task item and appends it to the global task queue.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_content: A dictionary encompassing the task's specifics and parameters.

    Returns:
        True if the task is successfully enqueued, False if it fails to be added.
    """
    task_item = _create_task_item(task_identifier, task_content)
    return _enqueue_task_item(task_queue, task_item)
