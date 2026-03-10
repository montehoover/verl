from queue import Queue
from typing import Any, Dict

task_queue = Queue()

def _validate_task_inputs(task_identifier: str, task_details: dict) -> bool:
    """
    Validates the inputs for a task.

    Args:
        task_identifier: The identifier for the task.
        task_details: The details of the task.

    Returns:
        True if inputs are valid, False otherwise.
    """
    if not isinstance(task_identifier, str) or not task_identifier:
        return False
    if not isinstance(task_details, dict):
        return False
    return True

def _prepare_task_for_queue(task_identifier: str, task_details: dict) -> Dict[str, Any]:
    """
    Prepares the task payload for queueing.

    Args:
        task_identifier: The identifier for the task.
        task_details: The details of the task.

    Returns:
        A dictionary representing the task to be enqueued.
    """
    return {
        "identifier": task_identifier,
        "details": task_details
    }

def _attempt_queue_task(queue_instance: Queue, task_payload: Dict[str, Any]) -> bool:
    """
    Attempts to add the prepared task payload to the given queue.

    Args:
        queue_instance: The queue to add the task to.
        task_payload: The task payload to enqueue.

    Returns:
        True if the task is successfully enqueued, False otherwise.
    """
    try:
        queue_instance.put(task_payload)
        return True
    except Exception:
        # In a real-world scenario, specific exceptions might be caught and logged.
        return False

def enqueue_task(task_identifier: str, task_details: dict) -> bool:
    """
    Appends new tasks to a global queue for subsequent processing.

    Args:
        task_identifier: A unique string identifier assigned to each task.
        task_details: A dictionary encompassing the task's specifics and parameters.

    Returns:
        True if the task is successfully enqueued, False if it fails to be added.
    """
    if not _validate_task_inputs(task_identifier, task_details):
        return False

    task_payload = _prepare_task_for_queue(task_identifier, task_details)

    return _attempt_queue_task(task_queue, task_payload)
